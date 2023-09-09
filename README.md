# XUND Data Science Technical Assessment

Natural Language Processing (NLP) of medical text is an important topic at XUND. Therefore, this
assessment concerns itself with implementing and reviewing code for Named Entity Recognition (NER).
But don't worry, this assessement doesn't require NLP experience, beyond looking up a bit of
vocabulary. Experience with Python programming and classification machine learning models will
suffice.

## Task 1: Data preprocessing

Some NLP models can only process a limited number of words at a time. Therefore,
the goal of this task is to implement `chunkify` in `task1.py`. `chunkify` should split text into
chunks if the text exceeds a given limit. The output of the function should fulfill the following
constraints:

 - Chunks must cover the entire text sequence and not overlap
 - Chunks must not be longer than the predefined limit
 - Chunks must not start or end inside entities
 - Each entity must be assigned to exactly one chunk

Beyond that, make sure that chunks are not too short i.e. if possible aim for larger chunks.
But it is not necessary to maximize the chunk size. Lastly, implement `chunkify` without
3rd party packages i.e. only standard library. You may however use `pytest` or other testing
frameworks to test `chunkify`.


## Task 2: Machine Learning

You are given the experiment found in `train_model.ipynb` which evaluates an NLP model (BERT) on a
NER task. The motivation behind the experiment was to find out if BERT works well for NER and should
be used by XUND in the future for NER problems (ignore the fact that we use TinyBert, this is only
for performance reasons).

The goal of this task is to review the experiment and find improvements with respect to
the experiment methodology and code quality. During the technical interview you will be asked to
suggest improvements to the experiment. Here are examples of how a suggestion may look like:

- The methodology used here is inappropriate because X. Instead you should do Y.
- Code in this cell is inefficient because X. Instead you should do Y.

*Note:* There is no need to write code yourself or run to even run the notebook
