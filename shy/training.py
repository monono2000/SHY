import torch
from pathlib import Path

if __package__:
    from .loss import *
    from .evaluation import evaluate_model
else:
    from loss import *
    from evaluation import evaluate_model


def _ensure_visit_lens_tensor(visit_lens):
    if torch.is_tensor(visit_lens):
        return visit_lens
    return torch.as_tensor(visit_lens)


def _iter_microbatches(patients, labels, visit_lens, micro_batch_size):
    batch_size = len(patients)
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        yield start, end, patients[start:end], labels[start:end], visit_lens[start:end]


def _weighted_average(total, count):
    return total / count if count > 0 else 0.0


def train(model, lrate, num_epoch, train_loader, test_loader, model_path, early_stop_range, objective_rates, device, micro_batch_size=None):
    model_path = Path(model_path)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    test_loss_per_epoch = []; train_average_loss_per_epoch = []; prediction_loss_per_epoch = []
    r2_list, r4_list, n2_list, n4_list = [], [], [], []
    for epoch in range(num_epoch):
        one_epoch_train_loss = []
        for i, (patients, labels, pids, visit_lens) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            batch_size = len(patients)
            effective_micro_batch_size = batch_size if micro_batch_size is None else max(1, min(micro_batch_size, batch_size))
            weighted_batch_loss = 0.0
            for start, end, mb_patients, mb_labels, mb_visit_lens in _iter_microbatches(patients, labels, visit_lens, effective_micro_batch_size):
                mb_patients = mb_patients.to(device, non_blocking=True)
                mb_labels = mb_labels.to(device, non_blocking=True)
                mb_visit_lens = _ensure_visit_lens_tensor(mb_visit_lens)
                pred, tp_list, recon_h_list, alphas = model(mb_patients, mb_visit_lens)
                loss, _, _ = shy_loss(pred, mb_labels, mb_patients, recon_h_list, tp_list, alphas, mb_visit_lens, objective_rates, device)
                micro_batch_weight = (end - start) / batch_size
                weighted_batch_loss += loss.item() * micro_batch_weight
                (loss * micro_batch_weight).backward()
            optimizer.step()
            one_epoch_train_loss.append(weighted_batch_loss)
        train_average_loss_per_epoch.append(sum(one_epoch_train_loss) / len(one_epoch_train_loss))
        print('Epoch: [{}/{}], Training Loss: {:.9f}'.format(epoch+1, num_epoch, train_average_loss_per_epoch[-1]))
        model.eval()
        total_test_examples = 0
        total_test_loss = 0.0
        total_prediction_loss = 0.0
        total_aux_losses = None
        metric_r2_sum = 0.0; metric_r4_sum = 0.0; metric_n2_sum = 0.0; metric_n4_sum = 0.0
        name_list = None
        with torch.inference_mode():
            for (test_patients, test_labels, test_pids, test_visit_lens) in test_loader:
                batch_size = len(test_patients)
                effective_micro_batch_size = batch_size if micro_batch_size is None else max(1, min(micro_batch_size, batch_size))
                for _, _, mb_patients, mb_labels, mb_visit_lens in _iter_microbatches(test_patients, test_labels, test_visit_lens, effective_micro_batch_size):
                    mb_patients = mb_patients.to(device, non_blocking=True)
                    mb_labels = mb_labels.to(device, non_blocking=True)
                    mb_visit_lens = _ensure_visit_lens_tensor(mb_visit_lens)
                    pred, tp_list, recon_h_list, alphas = model(mb_patients, mb_visit_lens)
                    test_loss, loss_list, name_list = shy_loss(pred, mb_labels, mb_patients, recon_h_list, tp_list, alphas, mb_visit_lens, objective_rates, device)
                    _, _, _, _, metric_r2, metric_n2, _, _, _, _, metric_r4, metric_n4, _, _, _, _, _, _, = evaluate_model(pred, mb_labels, 5, 10, 15, 20, 25, 30)
                    micro_batch_examples = pred.shape[0]
                    total_test_examples += micro_batch_examples
                    total_test_loss += test_loss.item() * micro_batch_examples
                    total_prediction_loss += loss_list[0].item() * micro_batch_examples
                    if total_aux_losses is None:
                        total_aux_losses = [0.0 for _ in loss_list]
                    for loss_idx, loss_value in enumerate(loss_list):
                        scalar = loss_value.item() if torch.is_tensor(loss_value) else float(loss_value)
                        total_aux_losses[loss_idx] += scalar * micro_batch_examples
                    metric_r2_sum += metric_r2 * micro_batch_examples
                    metric_r4_sum += metric_r4 * micro_batch_examples
                    metric_n2_sum += metric_n2 * micro_batch_examples
                    metric_n4_sum += metric_n4 * micro_batch_examples
        averaged_test_loss = _weighted_average(total_test_loss, total_test_examples)
        averaged_prediction_loss = _weighted_average(total_prediction_loss, total_test_examples)
        averaged_aux_losses = [_weighted_average(loss_sum, total_test_examples) for loss_sum in total_aux_losses]
        metric_r2 = _weighted_average(metric_r2_sum, total_test_examples)
        metric_r4 = _weighted_average(metric_r4_sum, total_test_examples)
        metric_n2 = _weighted_average(metric_n2_sum, total_test_examples)
        metric_n4 = _weighted_average(metric_n4_sum, total_test_examples)
        test_loss_per_epoch.append(averaged_test_loss)
        prediction_loss_per_epoch.append(averaged_prediction_loss)
        r2_list.append(metric_r2); r4_list.append(metric_r4); n2_list.append(metric_n2); n4_list.append(metric_n4)
        print('Test Epoch {}: {:.9f} (recall@10); {:.9f} (recall@20); {:.9f} (ndcg@10); {:.9f} (ndcg@20)'.format(epoch+1, metric_r2, metric_r4, metric_n2, metric_n4))
        if len(name_list) > 2:
            print('{}: {:.9f}; {}: {:.9f}; {}: {:.9f}; {}: {:.9f}'.format(name_list[0], averaged_aux_losses[0], name_list[1], averaged_aux_losses[1], name_list[2], averaged_aux_losses[2], name_list[3], averaged_aux_losses[3]))
        else:
            print('{}: {:.9f}; {}: {:.9f}'.format(name_list[0], averaged_aux_losses[0], name_list[1], averaged_aux_losses[1]))
        if epoch >= 30 and prediction_loss_per_epoch[-1] < min(prediction_loss_per_epoch[0:-1]):
           torch.save(model.state_dict(), model_path / f'shy_epoch_{epoch+1}.pth')
        early_stop = (-1) * early_stop_range
        last_loss = prediction_loss_per_epoch[early_stop:]
        if epoch >= 30 and sorted(last_loss) == last_loss:
           break
        model.train()
    return r2_list, r4_list, n2_list, n4_list, test_loss_per_epoch, train_average_loss_per_epoch, prediction_loss_per_epoch
