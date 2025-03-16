import os
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from model.config import load_config  
from model.M2EH import M2EH
from dataset.loader import load_data, load_checkpoint

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(model, filename=None):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


def load_pretrained(pretrained_model_filename, model):
    assert os.path.isfile(pretrained_model_filename), "Saved model file does not exist. Exiting."
    model = load_checkpoint(model, filename=pretrained_model_filename)
    return model


def save_roc_curve(all_targets, all_probs, filename='roc_curve_ed.png'):
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


def test_model(model, dataloaders, weight):
    print("\nRunning test...\n")
    model.eval()
    checkpoint = torch.load(weight, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    all_targets, all_preds, all_probs = [], [], []
    correct = 0

    for inputs, labels in dataloaders["test"]:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        output = model(inputs)
        _, preds = torch.max(output, 1)
        correct += torch.sum(preds == labels.data).item()
        all_targets.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()[:, 1])


    test_f1 = f1_score(all_targets, all_preds, average='macro')
    test_auc = roc_auc_score(all_targets, all_probs)
    test_acc = correct / len(dataloaders["test"].dataset)

    save_roc_curve(all_targets, all_probs)

    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    print(f'Test ACC: {test_acc:.4f}')


def main():
    import optparse
    parser = optparse.OptionParser("Test M2EH model.")
    parser.add_option("-d", "--dir", dest="dir", help="Data path.", default='./sample_train_data')
    parser.add_option("-w", "--weight", dest="weight", help="Saved model weight file path.", default='./weight/best_model_ed.pth')
    parser.add_option("-b", "--batch_size", dest="batch_size", help="Batch size.", default=16)

    options, _ = parser.parse_args()

    dir_path = options.dir
    weight = options.weight
    batch_size = int(options.batch_size) if options.batch_size else int(config["batch_size"])

    dataloaders, _ = load_data(dir_path, batch_size)

    model = M2EH(config)

    model = load_pretrained(weight, model)
    model.to(device)

    test_model(model, dataloaders, weight)


if __name__ == "__main__":
    main()
