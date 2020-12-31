#Emotion eeg signal

The data is in shape:

X: (session, channel, value)
Y: (session)

for example:
* X[0] is the first session. it's a list of list.
* X[0][0] is the first channel of first session. it's a list of numbers
* X[0][0][0] is a number

* Y[0] is a string (ussually) with the tag of first session