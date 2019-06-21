import torch


class Trainer:
    def __init__(self, config, model, optimizer, criterion, dataloader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = next(self.model.parameters()).device

        self.losses = MetricTracker()
        self.accs = MetricTracker()


    def train(self):
        for epoch in range(self.config.num_epochs):
            result = self._train_epoch(epoch)
            print('Epoch: [{0}]\t Avg Loss {loss:.4f}\t Avg Accuracy {acc:.3f}'.format(epoch, loss=result['loss'], acc=result['acc']))


    def _train_epoch(self, epoch_idx):
        self.model.train()

        self.losses.reset()
        self.accs.reset()


        for batch_idx, (docs, labels, doc_lengths, sent_lengths) in enumerate(self.dataloader):
            batch_size = labels.size(0)

            docs = docs.to(self.device)  # (batch_size, padded_doc_length, padded_sent_length)
            labels = labels.to(self.device)  # (batch_size)
            sent_lengths = sent_lengths.to(self.device)  # (batch_size, padded_doc_length)
            doc_lengths = doc_lengths.to(self.device)  # (batch_size)

            scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)  # (n_docs, n_classes), (n_docs, max_doc_len_in_batch, max_sent_len_in_batch), (n_docs, max_doc_len_in_batch)
            
            loss = self.criterion(scores, labels)

            loss.backward()

            self.optimizer.step()

            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # Compute accuracy
            predictions = scores.max(dim=1)[1]
            correct_predictions = torch.eq(predictions, labels).sum().item()
            acc = correct_predictions

            self.losses.update(loss.item(), batch_size)
            self.accs.update(acc, batch_size)

            print('Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f}(avg: {loss.avg:.4f})\t Acc {acc.val:.3f} (avg: {acc.avg:.3f})'.format(
                    epoch_idx, batch_idx, len(self.dataloader), loss=self.losses, acc=self.accs))

        log = {'loss': self.losses.avg, 'acc': self.accs.avg}
        return log


class MetricTracker(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, summed_val, n=1):
        self.val = summed_val / n
        self.sum += summed_val
        self.count += n
        self.avg = self.sum / self.count
