import torch


def accuracy(predictions, labels):
    return (predictions.argmax(1) == labels).sum().item() / len(labels)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, dataloader, batch_size, optimizer, loss_function, num_epochs, device='cpu'):
    model.train()
    model = model.to(device)

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    for epoch in range(num_epochs):
        loss_meter.reset()
        accuracy_meter.reset()
        for i, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_function(predictions, label)
            acc = accuracy(predictions, label)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), len(data))
            accuracy_meter.update(acc, len(data))
        print(f'Epoch {epoch+1} - loss {loss_meter.avg} - accuracy {accuracy_meter.avg}')


def test(model, dataloader, loss_function, device='cpu'):
    model.eval()
    model = model.to(device)

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)

            predictions = model(data)
            loss = loss_function(predictions, label)
            acc = accuracy(predictions, label)

            loss_meter.update(loss.item(), len(data))
            accuracy_meter.update(acc, len(data))

    print(f'Test - loss {loss_meter.avg} - accuracy {accuracy_meter.avg}')
    return loss_meter.avg, accuracy_meter.avg

