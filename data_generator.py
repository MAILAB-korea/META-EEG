import torch


def task_generator(config, data, label, iteration, mini_task_size):
    srcdat = data
    srclbl = label

    num_tasks = 7
    query_set_size = 576

    def create_tasks(srcdat, srclbl, num_tasks, query_set_size):
        total_samples = srcdat.shape[0]
        support_set_size = total_samples - query_set_size

        tasks_data = []
        tasks_labels = []

        for task_idx in range(num_tasks):
            start_idx = task_idx * query_set_size
            end_idx = start_idx + query_set_size

            query_set_data = srcdat[start_idx:end_idx]
            query_set_labels = srclbl[start_idx:end_idx]

            support_set_data = torch.cat([srcdat[:start_idx], srcdat[end_idx:]], dim=0)
            support_set_labels = torch.cat([srclbl[:start_idx], srclbl[end_idx:]], dim=0)

            indices = torch.randperm(support_set_data.size(0))
            shuffled_spt_data = support_set_data[indices]
            shuffled_spt_labels = support_set_labels[indices]

            task_data = (shuffled_spt_data, query_set_data)
            task_labels = (shuffled_spt_labels, query_set_labels)
            tasks_data.append(task_data)
            tasks_labels.append(task_labels)

        return tasks_data, tasks_labels

    def split_data(data, chunk_size):
        result = []
        current_idx = 0
        while current_idx < len(data):
            result.append(data[current_idx:current_idx + chunk_size])
            current_idx += chunk_size
        return result

    tasks_data, tasks_labels = create_tasks(srcdat=srcdat, srclbl=srclbl, num_tasks=num_tasks,
                                            query_set_size=query_set_size)
    mini_tasks_data = [(spt, task[1]) for task in tasks_data for spt in split_data(task[0], mini_task_size)]
    mini_tasks_label = [(spt, task[1]) for task in tasks_labels for spt in split_data(task[0], mini_task_size)]

    if iteration == 0:
        print("Task 1 - Support Set Data:")
        print(mini_tasks_data[0][0].shape)
        print("Task 1 - Query Set Data:")
        print(mini_tasks_data[0][1].shape)
        print("Task 1 - Support Set Labels:")
        print(mini_tasks_label[0][0].shape)
        print("Task 1 - Query Set Labels:")
        print(mini_tasks_label[0][1].shape)

    return [mini_tasks_data, mini_tasks_label]
