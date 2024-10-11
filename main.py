import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import *
from utilities import *
from torch.optim.lr_scheduler import LambdaLR
import argparse

# import nltk
# nltk.download('punkt')
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        #total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def CLS(tokenizer, rela_P, fname, L2= False, attM = False):
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=False)

    CLS_model = Classifier(
        numLayer=n_layer,
        maxL=block_size,
        num_heads=n_head,
        input_size=n_input,
        dim_model=n_input,
        hidden_size=n_hidden,
        dim_ff=100,
        num_classes=n_output,
        rela_P=rela_P
        ).to(device)
    CLS_optimizer = torch.optim.Adam(CLS_model.parameters(), lr=learning_rate,weight_decay=0.001) if L2 else torch.optim.Adam(CLS_model.parameters(), lr=learning_rate) 
    CLS_loss_f = torch.nn.CrossEntropyLoss()

    
    CLS_model.train()
    num_batches = len(train_CLS_loader)
    test_accs = []
     # for the classification  task, you will train for a fixed number of epochs like this:
    for epoch in range(epochs_CLS):
        CLS_train_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            pred = CLS_model(xb)
            loss = CLS_loss_f(pred,yb)
            CLS_train_loss += loss.item()

            CLS_optimizer.zero_grad()
            loss.backward()
            CLS_optimizer.step()
        average_train_loss = CLS_train_loss / num_batches
        test_acc = compute_classifier_accuracy(CLS_model,test_CLS_loader)
        test_accs.append(test_acc)
        print(f"epoch: {epoch}, loss: {average_train_loss}, test acc: {test_acc}")
        #scheduler.step()
    # plt.figure()
    # plt.plot(test_accs)
    # plt.title('Test Accuracy Across Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.savefig(f"{fname}.png")
    if attM:
        checker = Utilities(tokenizer=tokenizer,model=CLS_model)
        checker.sanity_check('The legitimate government of Kuwait will be restored to its rightful place, and Kuwait will once again be free.',block_size,f'{fname}_a')
        checker.sanity_check("It's time to put an end to micromanagement of foreign and security assistance programs—micromanagement that humiliates our friends and allies and hamstrings our diplomacy.",block_size,f'{fname}_b')
    return test_accs

def LM(tokenizer, attM = False):

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    with open("speechesdataset/test_LM_hbush.txt", 'r', encoding='utf-8') as f:
        test_LM_hbush = f.read()
    test_hbush_dataset = LanguageModelingDataset(tokenizer, test_LM_hbush,  block_size)
    test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)

    with open("speechesdataset/test_LM_obama.txt", 'r', encoding='utf-8') as f:
        test_LM_obama = f.read()
    test_obama_dataset = LanguageModelingDataset(tokenizer, test_LM_obama,  block_size)
    test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)

    with open("speechesdataset/test_LM_wbush.txt", 'r', encoding='utf-8') as f:
        test_LM_wbush = f.read()
    test_wbush_dataset = LanguageModelingDataset(tokenizer, test_LM_wbush,  block_size)
    test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)

    LM_model = TransformerDecoder(
        num_layers=n_layer,
        max_l=block_size,
        dim_in=n_input,
        dim_model=n_input,
        num_heads=n_head,
        dim_feedforward=100
    ).to(device)
    LM_optimizer = torch.optim.Adam(LM_model.parameters(), lr=learning_rate)
    LM_loss_f = torch.nn.CrossEntropyLoss()
    LM_model.train()

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        
        # LM training code here
        pred = LM_model(xb)
        loss = LM_loss_f(pred.transpose(2,1), yb)
        #print(loss)

        LM_optimizer.zero_grad()
        loss.backward()
        LM_optimizer.step()

        if (i+1)%100 == 0:
            train_perlexity = compute_perplexity(LM_model,train_LM_loader)
            hbush = compute_perplexity(LM_model,test_hbush_loader)
            obama = compute_perplexity(LM_model,test_obama_loader)
            wbush = compute_perplexity(LM_model,test_wbush_loader)
            print(f"train: {train_perlexity}, hbush: {hbush}, obama: {obama}, wbush: {wbush}")

    train_perlexity = compute_perplexity(LM_model,train_LM_loader)
    hbush = compute_perplexity(LM_model,test_hbush_loader)
    obama = compute_perplexity(LM_model,test_obama_loader)
    wbush = compute_perplexity(LM_model,test_wbush_loader)
    print(f"train: {train_perlexity}, hbush: {hbush}, obama: {obama}, wbush: {wbush}")
    if attM:
        checker = Utilities(tokenizer=tokenizer,model=LM_model)
        checker.sanity_check("Over the course of these 8 years, I've seen the hopeful faces of young graduates and our newest military officers.",block_size,'LM_A')
        checker.sanity_check("But I do have a couple of suggestions that you may find useful as you go out there and conquer the world.",block_size,'LM_B')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part1', '-p1', action='store_true', help='run part1')
    parser.add_argument('--part2', '-p2', action='store_true', help='run part2')
    parser.add_argument('--part3_achi', '-p3a', action='store_true', help='run part3 achitucture exploration')
    parser.add_argument('--part3_impro', '-p3b', action='store_true', help='run part3 improvement')
    parser.add_argument('--save_plt', '-plt', action='store_true', help='save attention map')

    args = parser.parse_args()

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)


    #CLS(tokenizer,True,'a')

    # part1
    if args.part1:
        print('part 1')
        CLS(tokenizer,False,'p1',attM=args.save_plt)
    if args.part2:
        print('part 2')
        LM(tokenizer,args.save_plt)
    if args.part3_achi:
        print('Architectural Exploration')
        CLS(tokenizer,True,'p3a',attM=args.save_plt)
    if args.part3_impro:
        print('Improvement')
        print('epoch = 40')
        global epochs_CLS, n_head, epochs_CLS
        epochs_CLS = 40
        CLS(tokenizer,False,'p3b_ep_40',attM=args.save_plt)
        print('epoch = 40, L2')
        CLS(tokenizer,False,'p3b_ep_L2',True,attM=args.save_plt)
        print('head = 4')
        n_head = 4
        epochs_CLS = 60
        CLS(tokenizer,False,'p3b_ep_head4',attM=args.save_plt)


    # plt.figure()
    # plt.plot(acc64,label = 'embedding=64')
    # plt.title('Test Accuracy Across Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig("Acc2embed.png")
    # LM(tokenizer)

    
  
    




if __name__ == "__main__":
    main()
