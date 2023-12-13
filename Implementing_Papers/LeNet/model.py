import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

class LeLightning(pl.LightningModule):
    def __init__(self, num_classes, learning_rate):
        super(LeLightning, self).__init__()
        self.lr = learning_rate
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.sm = torch.nn.Softmax(dim=1)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.c1 = torch.nn.Conv2d(1, 6, 5)
        self.s2 = torch.nn.AvgPool2d(2, 2)
        self.c3 = torch.nn.Conv2d(6, 16, 5)
        self.s4 = torch.nn.AvgPool2d(2,2)
        self.c5 = torch.nn.Conv2d(16, 120, 5)
        self.f6 = torch.nn.Linear(120, 84)
        self.f7 = torch.nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = F.tanh(self.c1(x))
        x = self.s2(x)
        x = F.tanh(self.c3(x))
        x = self.s4(x)
        x = F.tanh(self.c5(x))
        x = x.view(-1, 120)
        x = F.tanh(self.f6(x))
        x = self.f7(x)
        return x

    # Created this common step because this is a simpler problem where train, val, test is all the same
    # For more complex tasks you can define each of these differently!
    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(self.sm(logits), y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy},
        			  on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'logits': logits, 'y': y}

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(self.sm(logits), y)
        self.log_dict({'val_acc': accuracy, 'val_loss': loss})
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(self.sm(logits), y)
        self.log_dict({'test_acc': accuracy, 'test_loss': loss})
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        probs = self.sm(logits)
        preds = torch.argmax(probs)
        return preds

    def configure_optimizers(self): # And schedulers
        return torch.optim.SGD(self.parameters(), lr=self.lr)