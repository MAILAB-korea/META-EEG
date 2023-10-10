import os

import torch
import torch.nn as nn


def get_model(args, device, config, ti, current_path, load_classifier=False):

    print('-' * 30)
    print('Device:', device)
    print('Current cuda device:', str(args.gpu))
    print('Count of using GPUs:', torch.cuda.device_count())
    print('-' * 30)

    print("-" * 5, f"Backbone network : {config['backbone']} ", "-" * 5)
    print('-' * 30)
    print("*" * 40)

    if config['backbone'].lower() == 'deepconvnet':
        model = DeepConvNet(drop=config['train']['drop']).to(device)
    elif config['backbone'].lower() == 'eegnet':
        model = EEGNet().to(device)

    # """ Load pretrain classifier """
    if load_classifier:
        load_path = '{}/{}/{}/subject0{}/'.format(current_path, 'pretrain_model', config['backbone'], str(ti))

        classifier_filename = os.path.join('{}/subject0{}_classifier_model_state_dict.pt'.format(load_path, str(ti)))
        print(classifier_filename)
        model.classifier.load_state_dict(torch.load(classifier_filename))
        print("Load the pretrained classifier")

    return model


class DeepConvNet(nn.Module):
    def __init__(self, num_ch=22, drop=None):
        super(DeepConvNet, self).__init__()
        self.num_ch = num_ch
        if self.num_ch == 22:
            self.output_class = 4
        else:
            self.output_class = 2
        self.F = [25, 25, 50, 100, 200]
        self.T = 10
        self.P = 3
        self.sr = 250
        self.dr = 0.5
        self.num_blocks = 3
        if drop:
            dropout = drop
        else:
            dropout = 0.5

        self.features = nn.Sequential(nn.Conv2d(1, self.F[0], kernel_size=(1, 10), stride=(1, 1), padding=(0, 62)),
                                      nn.Conv2d(self.F[0], self.F[1], kernel_size=(self.num_ch, 1), stride=(1, 1), groups=1),
                                      nn.BatchNorm2d(self.F[1], track_running_stats=False),
                                      nn.ELU(),
                                      nn.MaxPool2d((1, self.P)),
                                      nn.Dropout(p=dropout, inplace=False),

                                      nn.Conv2d(25, 50, kernel_size=(1, 10), stride=(1, 1)),
                                      nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True,track_running_stats=False),
                                      nn.ELU(alpha=1.0),
                                      nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False),
                                      nn.Dropout(p=dropout, inplace=False),

                                      nn.Conv2d(50, 100, kernel_size=(1, 10), stride=(1, 1)),
                                      nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True,track_running_stats=False),
                                      nn.ELU(alpha=1.0),
                                      nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False),
                                      nn.Dropout(p=dropout, inplace=False),

                                      nn.Conv2d(100, 200, kernel_size=(1, 10), stride=(1, 1)),
                                      nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True,track_running_stats=False),
                                      nn.ELU(alpha=1.0),
                                      nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False))

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=2000, out_features=self.output_class, bias=True))

    def forward(self, x):
        outputs = self.features(x)
        outputs = self.classifier(outputs)
        return outputs


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self, num_ch=22):
        super(EEGNet, self).__init__()

        self.num_ch = num_ch
        if self.num_ch == 22:
            self.output_class = 4
        else:
            self.output_class = 2

        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 125), padding='same', groups=1),
                               nn.BatchNorm2d(8, track_running_stats=False),
                               Conv2dWithConstraint(in_channels=8, out_channels=8 * 2, kernel_size=(self.num_ch, 1), stride=(1, 1), groups=8),
                               nn.BatchNorm2d(8 * 2, track_running_stats=False),
                               nn.ELU(),
                               nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
                               nn.Conv2d(8 * 2, 32, kernel_size=(1, 16), groups=16, ),
                               nn.Conv2d(32, 32, kernel_size=(1, 1), groups=1),
                               nn.BatchNorm2d(32, track_running_stats=False),
                               nn.ELU(),
                               nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0))

        # Dense
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(32 * 33, self.output_class, max_norm=0.25))

    def forward(self, input):
        features = self.features(input)
        logits = self.classifier(features)

        return logits
