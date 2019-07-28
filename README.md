# COMBAT WOMBAT [4th place solution to the Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/leaderboard)

You should be able to replicate the solution and retrain all the models from [our inference kernel](https://www.kaggle.com/iezepov/wombat-inference-kernel) just by running all `train_*.py` scripts. One would need to put the input data and [the embeddings dataset](https://www.kaggle.com/iezepov/gensim-embeddings-dataset) to the `input` folder.

`code/toxic` contains various utils that are used in `train_*` files.

## Outline of our final solution

We ended up using a simple average ensemble of 33 models:
* 12 LSTM-based models
* 17 BERT models (base only models)
* 2 GPT2-based models

Not-standart things:
* We were predicting 18 targets with evey model
* We combined some of the target to get the final score. It was hurting AUC but was improving the target metric we care about.
* LSTMs were using char-level embeddings
* LSTMs were trained on different set of embeddings
* We mixed three different types of BERTs: cased, uncased and fine-tuned ucnased (we used [the standart fine-tuning procedure](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/lm_finetuning))
* We tried to unbias our models with some PL. The idea was taken from [this paper, "Bias Mitigation" section](http://www.aies-conference.com/wp-content/papers/main/AIES_2018_paper_9.pdf)
* GPT2 models were using a CNN-based classifier head isntead of the linear classifier head

## The loss

We trained all our models on a quite straightforward BCE loss. the only thing is that we had 18 targets.

* the main target, with special weights that highlites data points that mention identities
* the main taregt, again, but without any extra weights
* 5 toxicity types
* 9 main identites
* max value for any out of 9 identities columns
* binary column that indecates whether at least one identity was mentioned
  
All targets, except the last one, were used as soft targets - flaot values from 0 to 1, not hard binary 0/1 targets.

We used a somewhat common weights for the first loss. The toxicity subtypes were trained without any special weights. And the identites loss (last 12 targets) were trained with 0 weight for the NA identities. Even thought the NAs were treated like zeros we decided not to trust that information during the training. Can't say whether it worked though, didn't have time to properly run the ablation study.

In order to improve diversity of out ensemble we sometimes sligtly increased or decreased weight of the main part of the loss (`config.main_loss_weight`). 

The loss function we used:

```
def custom_loss(data, targets):
    bce_loss_1 = nn.BCEWithLogitsLoss(targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:7], targets[:, 2:8])
    bce_loss_3 = nn.BCEWithLogitsLoss(targets[:, 19:20])(data[:, 7:18], targets[:, 8:19])
    return config.main_loss_weight * bce_loss_1 + bce_loss_2 + bce_loss_3 / 4
```

The weights we used for the main part of the loss:

```
iden = train[IDENTITY_COLUMNS].fillna(0).values

weights = np.ones(len(train))
weights += (iden >= 0.5).any(1)
weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
weights /= weights.mean()  # Don't need to downscale the loss
```

And finally the identites targets and weights:

```
subgroup_target = np.hstack([
    (iden >= 0.5).any(axis=1, keepdims=True).astype(np.int),
    iden,
    iden.max(axis=1, keepdims=True),
])
sub_target_weigths = (
    ~train[IDENTITY_COLUMNS].isna().values.any(axis=1, keepdims=True)
).astype(np.int)
```

## LSTM-based models

### Architecture

We ended up using a very simple architecture with 2 bidirectional LSTM layers and a dense layer on top of that. You defenitely saw this architecture being copy-pased from one public kernel to another :) The only modificatrion we had is taking pooling not only from the top (second) LSTM layer, but from the first layer as well. Average pooling from both layers and max pooling from the last layer weere concatinaed before the dense classifiyng layers.

The diversity of oru 12 LSTM models comes from using different embeddings and different initial random states. We used the following embeddings (most of them you know and lvoe):

* glove.840B.300d (global vectors)
* crawl-300d-2M (fasttext)
* paragram_300_sl999 (paragram)
* GoogleNews-vectors-negative300 (word2vec)
* Char represnation. We computed top-100 most frequent chars in the data and then represented every token with counts on those chars. That's a poor man's char level modeling :)

Every LSTM model was taking 4 out of 5 concatinated embeddings. Thus the diversity.

We found the trick where you lookup various word forms in the mebdding dictionary being very helpful. That what we ended up using:

```
PORTER_STEMMER = PorterStemmer()
LANCASTER_STEMMER = LancasterStemmer()
SNOWBALL_STEMMER = SnowballStemmer("english")

def word_forms(word):
    yield word
    yield word.lower()
    yield word.upper()
    yield word.capitalize()
    yield PORTER_STEMMER.stem(word)
    yield LANCASTER_STEMMER.stem(word)
    yield SNOWBALL_STEMMER.stem(word)
    
def maybe_get_embedding(word, model):
    for form in word_forms(word):
        if form in model:
            return model[form]

    word = word.strip("-'")
    for form in word_forms(word):
        if form in model:
            return model[form]

    return None

def gensim_to_embedding_matrix(word2index, path):
    model = KeyedVectors.load(path, mmap="r")
    embedding_matrix = np.zeros(
        (max(word2index.values()) + 1, model.vector_size), 
        dtype=np.float32,
    )
    for word, i in word2index.items():
        maybe_embedding = maybe_get_embedding(word, model)
        if maybe_embedding is not None:
            embedding_matrix[i] = maybe_embedding
    return embedding_matrix
```

### Preprocessing

We did some preprocessing for the LSTM models and found it helpful:

* Replacements for some weird unicode chars like `\ufeff`
* Replacements for the "starred" words like `bit*h`
* Removing all uncide chars of the `Mn` category
* Replacing all Hebrew chars with the same Hebrew letter `א`. This trick grately reduces the number of distinct characters in the vocabulary. We didn't expect our model to learn toxicity in different languages anyway. But we don't want to lose information about that some Hebrew word was here
* The same logic applied to Arabic, Chineese and Japaneese chars
* Replacing emojies with their aliases and a special `EMJ` . token so the model could understand that an emoji was here.

## BERTs

### It's all about speed

We found that we can greatly increase the inference time for BERT models with two tricks. Both of them are possibly because of the `attention_mask` we ahve in berts. What it means for us is that we can zero-pad our sentences as much as we want, it doesn't matter for BERT. Wem managed to reduces inference time from 17 minutes as is to 3.5 minutes. Almost 5x times!

The first optimization is obvious - pad the whole dataset not to the some constant `MAX_LEN` value (we used 400 btw), but pad it to the longest comment in batch. Or rather clip the batch to remove some dead weight of zeros on the right. We did it like that:

```
def clip_to_max_len(batch):
    X, lengths = map(torch.stack, zip(*batch))
    max_len = torch.max(lengths).item()
    return X[:, :max_len]

# sequences is zero-padded numpy array
lengths = np.argmax(sequences == 0, axis=1)
lengths[lengths == 0] = sequences.shape[1]

sequences = torch.from_numpy(sequences)
lengths = torch.from_numpy(lengths)

test_dataset = data.TensorDataset(sequences, lengths)
test_loader = data.DataLoader(
    test_dataset,
    batch_size=128, 
    collate_fn=clip_to_max_len,
)
```

The second is much more important optimization is to sort the dataset by length. This allowed us to greatly reduce the amount of the dead weight of zeros. And just a tiny change to the code above:

```
ids = lengths.argsort(kind="stable")
inverse_ids = test.id.values[ids].argsort(kind="stable")

test_loader = data.DataLoader(
    data.Subset(test_dataset, ids),
    batch_size=128, 
    collate_fn=clip_to_max_len,
)

# preds = here_you_get_predictions_from_the_model
preds = preds[inverse_ids]
```

### Bias Mitigation with PL

We took and idea of adding extra non-toxic data that mentions identites to the dadset to kill the bias of the model. The flow is straightforward:

* Take an amazing datasdet of (8M senteces from wikipedia)[https://www.kaggle.com/mikeortman/wikipedia-sentences]
* Run predictions of all 18 targets on them with the best single model
* Filter out senteces that don't mention any identites
* Filter out senteces with high toxicity predictions (some movies and music albums ahve "toxic" titles like `Hang 'em high!`)
* Set the targets for toxicity and subtypes of toxicity to 0 since you know that sentecces from wikipedia should not be toxic
* Add those senteces to the training data to kill the bias towards some identites

According to the article mentioned above the trick works very well. We saw string improvemnts on the CV, which didn't translate to as strong improvent on the LB. We still believe that this is a very neat trick, but one needs to be careful with sampling - to math the distribution of lengths at the very elast, or better, other tokens.
