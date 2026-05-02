**Start HPC Experiments-20260421_183941-Meeting Transcript**

April 21, 2026, 5:39PM

2h 19m 6s

Sikar, Daniel** started transcription

Sikar, Daniel** 2:04  
Can I use that? Yeah, which I mean, is it? Is it? I don't know. Can I bring the right one? OK, I've got what you call it. I've got I think it's upstairs. I can bring downstairs. OK.  
Thank you.  
Yes, stay up.  
Okay.  
Yeah, absolutely.  
Put it back.  
Yeah.  
Reconfiguring site Tetris.  
Do we need to transcribe this? Yeah, it's transcribing. I think it is. Yeah. Thank you.  
Right, so you guys are new, so I'm just going to start from scratch and a bit about transformer architecture, how things work, so that you have context on what we're trying to do exactly. So, elements are essentially transformers, they do attention, they do multi-level attention. You remember I talked about attention?  
Attention is basically when you attention solve the problem of holding long-term context, which means that, let us say, when you're giving the model a data that has 900 sentences and there's a word in the beginning of those 900 sentences and the end of that paragraph.  
And somehow they are related. Models like LSTM, model before transformer, they would lose the context, they would forget that there existed that word by the time they read the last word. So if they were related somehow, the model would not remember. The transformers, because they put the whole string in.  
Imagine they put the whole string this way, and then this way, and then compare every word with every word, so none of the context was lost. That's what Transformer saw. Now, LLMs used that mechanism to predict and the issue, the problem with LLMs. I was the problem with the black box problem.  
And then, let's say this is our transformer model.  
There's an input that goes there.  
This is not what they come to.  
Okay.  
And in the midst of it, this so this text which is first tokenized.  
Then it becomes the book. By tokenize, we mean the message is a word. Like actually. Yes. If this was the sentence, I am British.  
There's a token that would be related to work. For example, let's say this.  
I think you said for a number that is obviously not that.  
But let's say, so this is what tokenization does. It makes our letters into tokens. So, by tokens, I mean, we get it again. Okay, and then this array becomes an input. And then we get an output, which will again be an array of tokens.  
And then we get boards out of it again. From this, all of this is element. This is what an element is. This is a transformer architecture.  
Now, what happens is that in the midst of all of this, like every other model, what we have are embeddings.  
So...  
Essentially, you, you know, we were talking about 62128 architecture of that, so these are all embeddings that are what are embeddings, the number, the list of numbers, something like this, but a lot more of bigger numbers, decimal points, all of that.  
So, the issue with the black box problem was that, once we put an input, for example...  
Gandhi's birthday. I don't know why I could have a better example. And it says 2nd October.  
This LLM would be trained on a lot of data, where there can be a lot of mentions of Gandhi, 2nd October, multiple things, but we don't know exactly where the model was able to predict that output from. We don't know that.  
But we, but our assumption, our hypothesis for the project that we're trying to work is that these embeddings.  
They must show some contextual information. They have to. That is why they are able to give us the output. They must show some contextual information from the input. And we are trying to validate it.  
So, what we did until we thought that now these inputs right here I talked about text, but now the models can take input text.  
Images and videos.  
We have not reached with you yet.  
We hope to finish images soon, but let's talk about text. Text is the one that we have worked on so far, and we have some results from so what we thought that we wanted to validate that this embedding should have context of the input that went.  
So, what we can do now, this input, the text that we can understand, that is humanly able, we know what it is, we can, you know, we can find out the contextual information, the facts, whatever there is in that text. So, for our experiment, we pick 6.  
Plus, emotions, sex emotions.  
And the text for me.  
So this sat.  
There is joy, there is love.  
This handy, this fear, and last one is surprise.  
This is the data set we made. Essentially, there are...  
About 1000, I don't know the exact number, about 1000 or something like that. I will get the numbers straight, about 1000 text sentences. These are sentences, so for sad.  
I had a terrible day.  
It's clear when a human reads it, it's clear that this is a sad text.  
Surprise could be.  
Ohh, no.  
Ohh.  
That was.  
So, something like this could be surprised. Now, these we knew we knew that which class the text would belong to, so what we did was we had a model like this, the transformer model. I say transformer all the time, because remember that when I look at this word...  
Or, let's see when I repeat this.  
Now, if I just see the word "new" might be positive, seems like it's a good thing, but if you look at the whole sentence, then we know that actually it's a surprise, so that's why we need to transform the model so that we get context from the whole sentence and talk word by word. We need the context of the whole sentence.  
So, what we do in the set of experiments is that we take this text, we go to the model, we take out the embeddings somewhere. Well, our experiments have taken it out somewhere at the end of it, but can be anywhere. Theoretically, it should apply everywhere.  
And this embedding in our text scenario is an array of size 768. So imagine there are 768 numbers, which are all decimal numbers, right? And then what we do is we imagine.  
A partition plane with 700 and sixty-eight axes, we point, we so these numbers, these are values to all the axes, and then we plot it there in that partition plane, and then we calculate the centroids.  
Of each class.  
So, by class, I mean that, all of this, you can do the centroids, the centroids would be the average of this, so let's say...  
Let's just say it's 2 numbers, and this was one, this was 1.5, so the centroid for these two would have the first one be 1 + 1.5 by simple that we take the average of 1 axis, all of the number, all of the number.  
Was in one axis for one class, the centroid will be there average, so we calculate the centroid of each class, and remember all of the calculations, the mathematics, has to be done in the 768 dimensions and not in the two dimensions that we just used to plot.  
We calculate, right? We also calculate the density. Do I have the access to the repository? Yes, so I'll share the screen and I'll put the repository there for you. So, just to clarify.  
The on the input side, it's uh...  
You have a data set with sentences, not like a large text, right? Data sentences still, so one data point is 1 sentence. Do you pass in a large bit of text, or do you pass in one for each? No, just one text. So if you were to pass a large text, you would probably divide it by sentences.  
Ohh.  
Yes, but right now what we're trying to do is we try to keep the experiment simple and we try to keep as less context as possible so to remove noise. So I don't want multiple sentences because in multiple sentences, the first sentence could have sad sentence, the second sentence could be happy, third could be surprise.  
that will jumble up our data point in the Cartesian tree. And the sentences in one by one. Yes, because that is exactly how our data set looks, by the way. So the results, we're looking at reports, right? Yes, actually.  
Are we looking at text model? I'll also join the meet and, or just pull the pull the plug there, and yeah, yeah, join the meeting, yeah, it'll be easier for me to explain.  
Oh, this was a multi-class derei emotions, I think, reports.  
Model selection first plan.  
No.  
Sing.

PG-Verma, Pritish Ranjan** 14:11  
Yeah.

Sikar, Daniel** 14:18  
So the meeting, I sent an updated invite. Okay, good.  
You're in OK.  
So, this is how the data set looks. Let me show you. We've done, so I'm talking about the six ones, because these are the ones where we ended up doing most of our calculations, but initially, if you also did a binary data set, let me show you both of them.  
So this was the first test. This is one data point.  
I'm feeling we have a problem, so I'm not very ambitious right now. This one data point labeled 0, which means negative.  
Let's look at a positive one.  
I left my, under my arm, feeling slightly more optimistic than when I, this is what you tried.  
There's no training, so you're using existing models. I'm using, yeah, you're passing this, I'm using existing models to pass it right now with the model things when it has not seen the data at work. There's just for the model, it's like imagine the right now, but I'm just passing a positive sentence to it.  
And then looking at them, looking at what do you think about it? We do, we do the training later to see what changes, but initially we just look at the existing model. You tell them, you tell the model that there it needs to be within these seven.  
No, the model is not predicting anything; the model is just encoding. So, again, good point. Let me tell you what encoding is. So, it's not always that we get text outputs, right? Sometimes what we do is you also talk about VLMs.  
which is visual language models here. What they do is they take image and then they text as well.  
So, imagine VLM would be there is another input image input.  
And how VLM works is that...  
It somewhere adds the information from image in the midst of it, in the midst of the whole process, and then together all of that is calculated, and then it gets down, right? That's all, so...  
You're just passing in this into the model. So, yeah, carry on, sorry. So, no, it's just, and then you grab a point of embedding, you grab it, and then you based on that, based on your embedding, based on that embedding, based on the centroid of that embedding and the shape.  
you can tell that it's this emotion. So I'll just rephrase that. And so the way I would explain this is just as you said. So the first question you asked, which was spot on, was like, oh, are we training it with that? No. So we get that phrase. We know what the label is because we have the labels there on the right.  
We get that phrase and we say, generate an embedding of this. It says, there's your embedding. We then label the embedding. Say, right, this embedding is our label. Move on to the next one. Generate an embedding for this. This is sad. You're generating an embedding of the sentence. Yes. And you're just, you're not asking for an output.  
No, just asking for where it fits within the model. Exactly. What is it? Imagine I fit the model. Does that make sense? Yeah. And then I just put in this. How do you know what point of the, I don't know, it's just context. How do you know what point you're taking the? The point at which this.  
This is a...  
So sorry, 700 is just a random number of dimensions of the embedding. So let's say if you were predicting MNIST, because they're 10 classes, the output of the network would be 10. So it would be a 10-dimensional space.  
You're taking the embedding from before the output layer? Yes. Is it right before the output layer? Sometime before the output layer. So essentially, we can...  
Usually, this, this every element has their own architecture. Yeah, right. So, somewhere, this would be, it's not fixed, but what's fixed is the fact that somewhere, I mean, embeddings can be, we could have worked with 1024 as well. But that's a good question, by the way.  
because that's a good question for the paper to say we've taken those embeddings out of this point of this architecture, because a referee could ask that question, where did that embedding come out of? So it would be a good thing to have like in the description of the methods, but it's logged here in the chat. So it should come up as an action point to say,  
that make sure that we, when we describe the experiment, we say that embedding came out of this layer of that transformer architecture. It's going to be in a transcription anyway, but I think that's a good point. Say, for example, obviously you're taking these predefined models, right? So you're taking...  
Gemini and Claude, right? Obviously, they're in the space somewhere in the middle of going through the process. They're not going to be the same amount of parameters in each space, are they? No. No. I imagine they wouldn't, but I don't know, to be honest.  
I imagine, but they could be.  
So like dimension to dimension, these embeddings will, that is almost like a time lapse of these, like how you get from point A to point B, right?  
What defines where you're taking the embedded from? What point of time? Is there a reason that you're taking the embedded from there and taking it from a random time at every different time? So, it's usually attention happening in the beginning layers of the model, right? By attention, it means that...  
Remember, the yeah, it's weights, so we don't want to miss it. Yeah, usually, even when we get CNN models, we make lots of CNN, and then the end of it, we put fully connected, we shrink it down to just...  
Single channel embeddings. By single channel, I mean type of stuff. Yes, just this. Usually, before that, it could be multiple channels. It could be, it could be connected, yes. So, you're thinking is a point of somewhere around here.  
How are you defining sunlight?  
So, the output models, model that we have outputs of, so it's a good question, but the thing is that we just taking 768 as a, because these models they usually have different output and various senses, right?  
Some the same model would have like 512 sizes, different sizes. Yeah, so I asked that question. So, look at this: this is a vision, this is the same clip clip, this is the vision encoder model. OK, same model will have this. See, this makes sense now, so there's different embedded sizes for each model.  
and you're taking it at the output embedded dimension before there is some kind of like function to spit out the class or like whatever it's doing in this. So our assumption is that we could have taken it at any point and the context should have remained there.  
Does, sorry, this is a this is a lack of my knowledge.  
You say you can take it at any point. Does that single point in embedding space, you're taking it from any point in the embedding space that is a single kind of like this, but not before that. If the input was sad, then the context of it being sad should be.  
So, you're taking it at the bare minimum at the point where it's reduced to one single dimension. Yes. OK, cool. That makes sense. Yes. And this could have been so usually what happens is that this is brought even smaller and, for example, if it were classification model, it would have been this 768 when he's talking about trying to implement this data, which is...  
Handed in zero, one, two, three numbers; we're trying to predict which number this is, so how do we do that? We get the output of any of size 10, which should have like 0, the index, which is for zero, would have the maximum number.  
That's the process.  
Do you have another pen? I have another pen. Yeah, it's upstairs. I'll be back in 5 minutes. Just run out.  
This one.  
I explained. I will get pens until you meet.  
So, we've tested this with several different models.  
Is there any point where we're not taking it from 768? No, for uniform and following that, usually change only one parameter when you're testing things, because, so I know that if something has changed, it is because of that parameter, so I want you to keep 768.  
So, all of the models we're using have an output embedding dimension at some point that is equal to 78. It doesn't matter where a lot of it is. It might be that there's one after that's smaller, but it will all of them will have 76 and 76 is for some arbitrary number that maybe.  
In computer science, usually they create something they should.  
Say that again. In computer science, usually numbers that they're considered their powers of two, yeah, like, yeah, yeah, because everything's binary, so it's easier, so they don't of the computer science won't use 1000, they use 104, yeah, because that's a, and that's just one of those numbers, yes, OK, cool, that makes sense.  
It's not good skills.  
512 is, you know the 768 dimensions. So, so where are we? Yes, we take these embeddings, we plot them out in the 768 dimensions. Wait, are you clear about the embeddings or do you? I understand, I understand everything. I'm asking just why, but I understand what's going on.  
When you, your, because 768 is some like an embedded dimension that exists in all of them, you can fix it at that point and plot it out at that point. 768 is just the size that we have taken. 512 would not have been wrong. 1024 would not have been wrong. But the thing is that if I take 1024 once, then I have to make sure that I use it to  
But like if you took 9999, for example, that doesn't exist in one. That doesn't exist. So, one is usually, yeah, they have outputs, so we take one of them. You're plotting the embedding at a fixed.768 that you know exists in every single model. And that plot, when you do PCA,  
it reduces to a roughly the same shape, which is this kind of donor. And from that, this is the whole, this is the whole thing. Yeah. So you're talking about the complete data, but let's just talk, let's just stick to one data point right now. So what happens there is that this, let's look at one of the next.  
So, this was maybe also show you that this was binary data.  
This is multiple emotion data.  
The  
Yes, so this, this is the data that we're essentially working on in the most.  
This will have a text and the emotion tag with it, right? The clean text is just removing the punctuations and everything, so, and then for one data point, one of these sentences posed, my biggest fear is coming through right now, apparently. That text tokenized goes to the model.  
We take, we take this bedding in the middle of the model. It's different, it's not say mix, it is somewhere near the end of it, but for, but because there can be other calculations after the 768 dimensions is left there, so we see.  
And then we block over that one data point there, so this my biggest fear is coming through right now, just if I show you in the...  
So, blocks then.  
It will be one of these purple clouds.  
That sentence becomes one book window.  
And then what we do, so we brought it back, we did PCA, we brought it down, the centroids are calculated by taking coverage of the three point in the 768 dimensions, and then even the centroid is blocked out in the dimension. So none of the mathematical calculations is done in, it's done before.  
It's done right now. We want to make sure that all of the calculation is done in the 768 dimensions, and then the two dimensions, it's just to visualize, because we can't visualize, and each of these dots is a sentence. Each of these dots is a sentence, exactly. Every single dot is a sentence.  
And the color of the dot tells you which class it belonged to, before it will became.  
Great.  
And our assumption was that, when we look at these points, all of the points, we thought that the sentence that is most scary, most fearful, would be near, you know, near the centroid, it peaks at the centre, and then gradually...  
It very hit spots.  
But turns out that...  
So we thought it was density, it was not density. Now imagine that this is not density, it's a number of data points per radial distance back. What I mean by that is that this is a centroid. Let's say there's a circle here at distance 5.  
and there's another circle here at distance 6. We're counting the number of dots between these two circles, so this pan right here, we're counting them and then that's what this is. There's the number of points between those two circles and that group, that group over and then came down.  
But then I brought, then I think there was a band here, and a band here.  
This band is way bigger than this band, so obviously you have more data points. Does it make sense? Yeah, so it growing makes sense then, because initially I thought that the density group, but density does not show the growth. It's the number of data points per really distant band. Does it make sense? Yeah.  
So that that, yeah, yeah, because now in the bigger back there can be more, you can fill in more dots, that's why there were more dots, so it grew and then it came back. So this, the word density here, we are going to change it to number of points per radian distance. Yeah.  
And you expected it to be like exponential this way.  
Which is what happens, so this is this is when we take the absolute density. By that I mean we divide it by the volume, so whatever possible number of data points they can be, we divide it by it, we take, so if this can hold 10 values and it has nine values, it's 0.9. If this can hold 5 values and it has five values, then the density is 1, so the density is...  
More there, that makes sense, and then we saw exactly what we thought would happen: that the density is at peak near the center, and then it goes down. But one thing to note here is that the density starts right after seven, so this is nothing from the centroid.  
To a distance of seven.  
So, remember, we don't work this, and then there's a hollow, there's a hollow area between any four. There's nothing that completely represents fear, completely represent fear. It's the way it starts a certain distance, and then, yes, I would back check, so go on.  
No, no worries. Okay, so before you argument that doing sentences is better than long text because it kind of compresses the data, it will be better for quality of the data, but how do you determine whether the sentence itself has like unnecessary context, right?  
Like a fiddler works or like the and.  
So, because it's already done in the hashing, I assume. No, we don't remove the words like "the" and all that, because they hold context. Yes, happy and not happy, something. So, if we remove words like "the", let's say we remove something like "not" or...  
Isn't.  
So, you know, isn't happy and happy. If we remove something like isn't, then they're both happy. But isn't happy is a negative word. I'm just thinking about there will be a difference between a short test sentence and a very long sentence. Yes, but we, and if we were talking about that, then we're trusting the data set.  
The trusting that it doesn't let me have history, right, and...  
That there, there is not a lot of noise there, and as a reader.  
There's not irrelevant information in those texts. If a sentence says something and is tagged with an emotion, that when resuming that that sentence talks about that emotion and there's not, you know, no fact there, it is displaying some emotion.  
Because we have to trust the data set to start the experiments. That's another interesting point, right? If the labels were noisy.  
I know. But that's like the future thing. Yeah, I mean, I know, I know that. I thought of that. I thought of the model that we were there to create that confidence for, and then we put a threshold on that confidence for, but that's the way to start. Yes. Not now, definitely. I thought of a way to remove noise.  
So, yeah, I forgot where we go. Yeah, that's right. So, yes, there's nothing.  
There's nothing there in the distance of seven, and then after that, the density goes down, and then we look at this gap, we look at this.  
And we thought that maybe this could be, because also one thing to note, although both of us, surely both of you will have repeat for the signal.  
Positive emotions tend to be closer, the centroids tend to be closer, and the negative emotions tend to be closer to the negative ones. They stay far away from each other. Something like surprise, which is a reaction to those emotions, so can be aligned with multiple emotions, is somewhere in the middle. It's not.  
Well, I mean, in two dimensions, we see it in the middle, but could be that you know these points, there is a cluster, a cluster has surprises somewhere there, but when you bring it to two dimensions, you're looking at it in the middle of it, so, but we assume we have calculated the distance.  
Class distance.  
Out of work, in anyways.  
So, this, this was the first set of experiment that this is the plot that we got after, and this can this can conclude few things: first is that there is no pure emotions, second thing that in the data points of 1 emotion do form a cluster.  
You really?  
The noisy cluster, or sorry, not noisy, it's a wide cluster, it's not very tightly backed cluster, but they do form a cluster. And also that there are overlaps. Now, one thing about overlaps, so when do I say that?  
Something is overlapped, I'll show you that.  
So, the circled ones here are overlaps. Why are they overlaps? Because this, if we, this data point right here, this is an overlap because although it is assigning the color orange, this point is closer to the blue center.  
So, if a point in those dimensions, 768 dimensions, is closer to another centroid than its own centroid, then we call it an overlap. OK, make sense? That is the definition of overlap.  
And we saw that, I mean, now, as right after the number of data points per radial distance band reaches its speed, after that, the overlap increases. So imagine when I'm talking about the band that I was talking about. So this is the centroid of joy. There's a band of joy. I'm saying that right after this. Isn't that interesting? There's a band of joy.  
Imagine that written in the paper. There's this band of joy. There's no pew.  
So, what this concludes is that this band include this band would not end.  
This after this band of overlap occurs, so before that the overlap, the chances of overlap is minimal. I'll show you this way.  
She, here are not very opposite emotions.  
Yes.  
So, you see how early are there? I mean, the data set can be a little noisy, so this can be when we have like a proper confidence data set, but right now, even in this, the overlap between before this one is very minimal, but it peaks around like after 9.5, we can say the overlap has started. Let's look at the density.  
I drink that.  
Not between 9 and 9.5.  
399.5 is where it started to be. So the overlaps always start to happen majorly after this ban. This is another thing that we found out.  
Ohh.  
Right, so this, this was the this was all that we found out before from text until now, and then we thought that, OK, this was a pre-trained model, a kind of model that has general context of everything. Now, we found that, let's try to train a model on this.  
data set. So, for right before that, one thing that I missed was that our assumption, there's another hypothesis that we claimed at this point that something that an emotion, we found out that, you know, there's no pure emotion or the emotion will be a combination of emotions.  
Because, if something, if we are feel, if we are sad, sad out of handliness, then this the point should be between sad and I, does that make sense? Yeah, so...  
So, our next hypothesis was that...  
Oh, and there's a there's a thing as well.  
I hope it works. Yes, it does.  
Just to abstract real quick, you can confirm that the data set has sentences that are non-neutral.  
It's like, for example, there is a cup on the table, so that that would have carry zero adoption. Are there like sentences? Exactly. Now, that is that is the thing that was the data set. We have to trust the data set at this point, and then we can do the filtering of data sets, so that is why this looks very noisy right now.  
But what might, what another thing that we could have done to remove that is like ask a model to give a confidence score. So let's say this is a sentence, is it sad? How sad is it? And if it doesn't, if everything gives a number of 90%, then only we put it in the data set.  
But it was just the two of us working, we didn't have time to do it. Scope is exponentially growing. OK, we have an answer for where the embedding is taken.  
So, and I will just ask the guy to do it, to put it in a read me in the text model next to train underscore multi-class top high, because that's where the multi-class training script is. And basically it's saying the actual extraction happens at line 7476 when the pattern buildings are taken from the last hidden state.  
and mean pulled over tokens. So the answer is training job path. Embeddings are taken at output start hidden state block, embedding generation path, which is the state that we're looking at here, embeddings are taken from model output last hidden state, then mean pulled, so at last hidden state.  
That makes more sense to me. That makes a lot more sense.  
Yes, ohh, wait, that's a wait, somewhere there, yeah.  
OK, so now, next set of experiments. Now, we know that one thing that we want to next hypothesis is that if something is sad and fear, then it should be somewhere inside and fear. Now, remember I was talking about how classification model output looks like?  
This, if we're trying to predict between dog and cat, that's a simple, we may try to predict between dog and cat. Now, when we train...  
This is how we tell the model. This is the ground rule for the model. So the model doesn't know the word dog or the cat. It knows that one set of images should give this output and another set of images should give this output. Right? Does that make sense? And if we, if we're talking about emotions, then it would...  
Become something like...  
Yeah.  
Does it make sense? Yeah, more like a like a softmax or something? Yes, yeah, we call it one-one numbers. OK.  
Because only one of them would be one, and all of the others are zero.  
But how do models work? Models don't.  
You know, in fact, it gives you the probability of something. How probable is that this is this? So, even though this is how the groundwork looks, when the model is getting on output, usually it will be like this.  
Yeah, yeah, so it's it's again, if my model is very accurate, then it can even it can reach almost zero, but you know it's still probabilities, yeah, we do benefit of that, so, and then this is after softmax, you know what softmax is, yeah?  
Yes, so, softmax is essentially this was zero point. Now, before softmax, this is also process model output can be 0.8 and 0.7. What softmax does is that, because they're independent probabilities, independent probabilities, that means when the model was trying to predict this.  
It did account that this could also be true, otherwise.  
Your probabilities have to sum up to one, probabilities of all the cases. So, what Softmax does is it does this into something like 0.6 or 0.4. I mean, calculation is 0.8 by 1.5, sum of this in 0.7 by 1.5.  
So, something like this.  
So, this is the softmax is this, right? But before softmax, these are independent probabilities. That means when I'm trying to predict the value of this, or when I'm trying to predict this...  
I'm not really concerned about this, although it's somewhere in the modification effort, and now this is how probable.  
Is the image dog?  
And this is how probable is the image cat, right? So, I'm hoping that when we're talking about 6 emotions.  
And we look at individual probabilities before soft maps, and if some sentence is a mix of emotion, then the problem needs to look like, let's say, if this was sad and this was fear, and I'm standing, the emotion is a mix of these two. Sometimes 0.4, 0.4, and everything else is 0. something.  
Yeah, small numbers, essentially, right? So, our assumption that, and by the way, the softmax is done after that, so these are independent probabilities, yeah.  
So, our assumption was that this point, 0.4 this, this is 0.45 this.  
If fear is 0.45, an assumption that it will be between fear and sadness, and a little bit closer to fear, yeah, yeah, yeah.  
So.  
So, we got to train the body and see what it does, so...  
The other side group.  
Better that the other side of this punch we do better on then.  
Ohh, state.  
No.  
Yes.  
So, we thought that, OK, let's look at this architecture, try to recreate it, but essentially there's an important data.  
We make a custom one.  
Custom by this is for first time we use quen quen is the transformer that we use, and what we do is we take a point last hidden state for somebody closer to where the model and embedding of 768 right? OK, we take this and.  
Seven 68, we do fully connected layers. By that I mean that we bring down the we can bring down the single embedded channel, keeping it single, single embedded, but we can bring the size up or down. We bring it down to six.  
Six emotions. Oh, I see. Yeah. Six. Yeah. So then the 768 six and all of this becomes one one.  
This is a custom model. This is something that. Yeah, so you're defining the output stages. You're bringing it out to six, hopefully 6 mirrors your prediction of everything that's customer. So this was a pre-trained transformer-based model. We say, just give us your last time and say it was the site of 760.  
And we do some calculations that that should be a part that should become a part of the model. So the way you do this will also be trained. These parameters that bring down the model size from 768 to 6, they also get trained and they will be changed and affected by how the data set is.  
So, our thought process was that also, okay, remember coming back to this, we were talking about how the we don't know what's happening in the middle, but we don't know where the context is.  
But, my son, if they intended to start, it would be sad to open it, it would be sad to see it, and it would be sad at six.  
So, we thought that when we could look at the output of 6.  
And we plot this 768 in the graph. So this 768 is plotted this way there. And then we look at the output of this, let's say 0.9 of a SAT, and everything is a small number. Then if one input, if one sentence.  
This goes there, at one point in the model, it is the 768, we take it out. At the final layer, it is the output of 6, we take it out. The 768, we plot it.  
Yes, yeah. And then with six, we're okay.  
And then our assumption right now, a hypothesis that we're trying to prove is that this...  
And this, there should be a mathematical relation or some sort of relation between how this looks in the 768 dimension and the one.  
They should be correlated.  
It makes sense, because I mean, all of this was just one model, then it's just one model, we're just looking at, we're just looking, we're just making, it's just compression, so they'll be the same and the mathematical difference is the act of compression. Yes, mathematical act of compression.  
But we're saying that, if we in that graph and we look at these six times error, there should be some mathematical correlation between, yes, yeah, you know our assumption that if something is 0.4 and 0.4 for fear and sad, then it should be somewhere between fear and sad, yeah, I'm just trying to prove.  
So, do you know what I was thinking just now, British, that if we, up to that point, we still don't look at the labels. We say still at that point, we know what the label is, but what we're going to do is plot that in the six-dimensional space and say what clusters form in six dimensions.  
And I guess that could be a way of validating this, of saying, well, in the 768 dimensional space, we know that all anger falls within. And then in the six dimensional space, because we know what the label is, we say, keep plotting that. And then we'd see the six clusters and we'd be able to distinguish and say,  
There's a surprise, there's the anger, there's a happy, there's a sad.  
What's that sound like? I mean, how you can bring it down to six times, you can be done even C6 times.  
No, but we would get the average centroids.  
like we get with this thing, and we could say, well, the average centroid for the every time we put anger in there, we see that this column is spiking and the other five are not something in those lines. Every time we throw happy in there,  
We see that that column there is spiking and the other five are not. If we take all the averages of this, the spikes that we believe are anger, and then we say, okay, there's a cluster forming here where anger is. And if we take a cluster of every other one, of an average, all of the other ones,  
we should see clusters in six dimensions, I imagine. Yes. So let's say we have that set of results and we should have them already. We should have a number of these results that has been running on the HPC since last week. So then we have these outputs here, which are  
And I think they're logits, right? They're not softmax. I don't think we're normalizing it. So where do we take those results again? So, I mean, they should correlate, right? We should find a correlation between what the scores we're getting from the logits.  
and the clusters we're seeing in the last hidden states.  
Right? That's the assumption. That's the assumption. So, right. OK. Yes.  
I mean.  
Yeah, go ahead for a second.  
Your.  
From what I understand is, you're mathematically defining.  
from 768 to 6, what is the relationship there? Are you making the assumption, are you choosing 6 because there are already 6 emotions? Is that why you've chosen 6 as the outlook? Yes. OK. So your assumption is that.  
by the time it gets to this stage of the transform, well, this isn't past the transform, it's the second one, the whole one as a whole. When it gets to this point, it will have distilled all of the information to six emotions.  
I'd say that's the assumption. Is that the assumption? That would be my assumption, yes. Just to clarify as well, you're just throwing these sentences in the model, right? You're not telling it to do anything. Are you telling it to do anything? You're not telling it to do anything.  
Models to analyze images, so it's, so you're not giving the model an objective to, nothing, no, no, so you're saying that naturally just passing in a single sentence the model's output from 768 when you 768 and six will be wrong.  
One thing to fix that when we train the model, yeah, this is at this stage when we're training the model with every action we are training it to predict that if the sentence was sad, then you should be able to predict something. Sorry, so here we're training this.  
this batch of experiments, we're actually training the model with, of course, 10, 25, 50, 100, 1000 epochs. Sorry, Josh, I'm going to take that back. So in this guy, in this case, we are training the model. So whatever happened there, like the fully connected layers, we're adjusting those weights to learn that data.  
Yes. Six emotions. So, yes, it has to learn that this segments belongs to one. Yeah, because you're doing it has a right. So, yes, initially in your initial, it did not, it did not, and so at some point when you just took up by through experimentation.  
You took out the embedding in 762 and you saw that these patterns existed, and therefore you thought. When you had embedded the model, it just, you just put it in and you saw the space of these and you were like, okay, well, at this point, the patterns are similar and there's only six of them.  
And therefore, at this point, we know that it's trying to work out an emotion. Now, you're training it to do the same thing by adding an extra, extra, like couple of layers to the end of the model to do this specific thing. Yes, and training it to predict the emotion. So now you're proving, so you've got  
You've got your observation, and now this is the proof that this 768 dimensions, when it's distilled into six, they are the same thing. That's what the proof is here. So the proof, the thing that we're trying to prove here, is that in any of the one, this architecture or this architecture, you imagine the last minute,  
Is output of the model? Yeah, so this is what the model said.  
Yeah, and everything before that is what the model thought.  
I'm just trying to put good with this, yeah, yeah, imagine it as a human being, yeah, everything before it said something was good thinking, yeah, so by that logic, this 768 is a thought of the model, yeah, and this six is what the model said, we're trying to correlate what the model said.  
With what the model is hotter?  
Okay.  
And that's fine because you're building the second part of the to specifically see what it's thinking. Essentially, I didn't tell the model to say anything. Yeah. I said, what do you think of it? Yeah. Without training, without, I didn't ask you to say anything. What do you think of this?  
Yeah, this is what came up now. I'm like, "OK, model, you're supposed to learn how to predict, how to classify these sentences in one of these six emotions, and we train the model, and now what now when you put the sentence, it knows that what I'm supposed to say is which class."  
Does the sentence belong to? It's still thinking, though, you're taking at the last point before it speaks, before it literally speaks, you're taking the embeddings, and you're taking that embedded and building a model to classify it into six based on this five or four, and the plot is, sorry, I'm...  
The plot itself, right? At 700 and sixty-eight dimensions, it could be saying any number of things. It says this was bad. There was no objective to it, so it could like that, that's what the model had no objective of actually classifying it. It just looked at the sentence and then caught something.  
Okay, so you've caught it. Yeah, that makes sense. The only reason I was confused is because, to say, for example, you pass it, I had an apple, it was bad, right? Apple still has some meaning there. Apple exists in 768 dimensions, but the pattern at the end of the model has always produced emotion. Emotion is the last pattern before we get to that. You're absolutely right. So in the pre-trained model,  
We did not tell the objective of the model is emotions. So in this particular plot, the graph, the 768 dimensions won't just hold context of an emotion, it will hold context of everything in the sentence that could be there. So when we look at the data, it just so happened that the pattern that emerged is  
I really like being around him right now. It belongs to the class love, but when a pre-trained model looks at it, is this the raw data set? Yes, this is the raw data set. So it has labels already.  
Yes, I did a box in the like, yeah, yeah, yes, so, so it's a form of unsupervised training.  
Quick.  
So, the first experiment where we get the output of that of the last hidden states is to say, if we get one of those phrases that is labeled as anger and look at the embeddings that were generated, where does that  
Point 4 in this 768 dimensional space, and it says falls over there. And we repeat the experiment for every one of those things. So, because we know the labels, and then we go and say, look at a plot and say, where did the can you show us some clusters here? And it says, well, it so happens that all the  
the anger phrases fell over here, and then the, what's the other for anger, fear. In fear, if we look at the centroids, we have the embeddings and we can say, take the average of every embedding for anger. Where is it in this 768 dimensional space? It's over here on the left.  
768, yeah. We're averaging and there might be a couple of 1000 of each or 1000. Where does that average fall in this 10-dimensional space? It falls over here on the left. What about fear? Where does it fall? Same exercise again. We didn't tell the model anything. We just gave it this phrase.  
that is labeled as fear, and we say, where is that thing? And it says, it's over here, and it so happens to be close to anger. And it's not doing it in relation to other sentences, it's still independent.  
Yes, did you, did you?  
When you plotted it and you took it, the PCA, whatever, it got down to a point, when you took a centroid of, you took all of the sentences specifically related to anger that were already labeled and you found the centroid of those labels. And that's your now, that's your new definition of this emotion.  
Right, also, you pointed out one thing that, before we were given any objective to the model, the model will try to then say, "I gave it the sentence: 'My sister needs to learn right now.' He asked me what you think of this pre-trained model, the one that helped us not this, yeah, right?"  
back to the sentence now. This model, when looked at this sentence, it did not know that it was supposed to act on emotions. So it will hold context of everything, even the fact that system is being mentioned. Yeah, because obviously if I put it out to a model, it might spout.  
Well, this is something like that, right? It has no objective right now, so it trackable context of everything. Maybe, Daniel, that's also why the clusters were a lot wider, and after training they got tighter, because initially those embeddings held context of everything in that sentence.  
Not just emotions, everything. The reason...  
So, sorry, carry on, Josh. Go the reason. So, my is just, so it's just interesting that, from from what you said earlier, you have always taken it at the point before the output, right?  
And plotting that, it seems like, based on the clusters, just based on the clusters of plotting in 768 dimensions, it seems like the last stage of the model is emotion, like the last stage of the model. Do you understand what I'm saying? Because if it's clustering like this,  
That means the most important has been put on emotion, even though you didn't tell it to put it on emotion. The most important, the most attention is being paid to emotion if that's not true. Is that not true? You are absolutely right. These embeddings won't just have context of emotions, they will have context of everything. Before being trained,  
If there's a point right here, thank you, that that fixes a lot, that's now we know my training. Let's say if there's a point right here, and this was so this point right here was just hold context of the emotion, but hold context of everything.  
So in that sentence, there's a context that sister is being mentioned in sentence.  
Right now, there's a context about time that it that it talks about presence, so maybe that's why there was a little sort of logo labs, things like that, because the context.  
It was context of everything. That's an interesting discussion. Actually, sorry, but actually right now doesn't make sense 'cause we don't have the reference for it, so...  
Yes, because, yes, initially, that's a good point, but...  
There's no time, so you can do some things. I got, I have a second question, right?  
Now, you've given 7 labels here, right? And you found the centroid of those labels. Now, say there was something else that had seven possible classes, like days of the week, right? If you plotted the centroids based on the labels of days of the week for this, for example, for any number of things,  
Is what like does the pattern, I know this is separate from what you're doing, but what pattern is exists? Because then it's more like, is it defensible that this pattern is just emotion? Do you know what I mean? Yeah, no, that's why these images we're trying to do for things that are absolute. But let's think about the days of the week experiment and we go from Monday to Sunday.  
And there's something in the phrase that says something about the day of the week. Oh, today is the day of the Lord. And then it's like, oh, it's Sunday. And oh, God, every day is like, dot, dot, dot. And then it's like, oh, that must be Monday. Tomorrow's Sunday. That means there's a context for Saturday. Yeah, for Saturday, right?  
It was an example of how what you've done is you've given labels and found centroids based on this huge dimensional, like, space, and is that is that replicatable with something else that is also certain labels, because this is just a cloud of thought that the LLM is having, and the thought exists in this space. Do you know what I mean?  
But that's it's completely irrelevant to what you're doing now. It's just that it's not, but that's true, but that's true. We'll have to. I mean, the hope is that 7 clusters would form, like Monday to Sunday, and say Monday is over there, Sunday is over there, and so forth. And if someone says today is tomorrow is Sunday, then it should.  
We close a little bit Saturday as well. OK, so it it your your it's just clustering the clusters themselves represent the thought and you're trying to get the information and the context out of the cluster itself. Ohh, and let's this is just kind of going way off piece, but imagine if you have like...  
the layer that had the days and be thrown in there, there was emotion as well. And the expectation was that happiness is likely to cluster near the weekend and sadness is likely to cluster at the beginning of the week, that kind of thing. But that's the kind of way we're going, right? It's like, how do you kind of... You're doing it at the most base level and you're working on it.  
OK, right, so, yes, so, so, so now we this was before the model had a new objective, it'll have context of everything you just pointed.  
Now, we give this model training, which means we have given it an objective. Hey, learn how to predict an emotion from simple. Now, this model knows what the task is, so it will be thinking about the context that it has to take. It will be thinking that now, OK, now my purpose is to read emotions, so it will hold more context about the emotions.  
So, that way, these my assumption a little bit by training that these clusters could tighten, move closer to the to the, which is exactly what happened. Oh my God, you have those results now? Yeah. Oh my, for how many epochs? You push one of the trainings.  
Which was 10, I think, balanced 10. Yeah, go for it.  
Wow.  
SSH Liberia.  
I really think.  
Cool.  
Go, M.C.P.  
So, now we give the model, we trained this model to when I was talking about 768, 6, right now only block the 768, so in a sense that...  
Let's not look at the output of the model right now. Let's just look at what the model thought the embeddings.  
See how clusters are getting that. That's why, by the way, this is because this is a different model. I thought that I'd ask you, can you also pre-train model and generate embeddings on the test set so that we can compare what the actual changes between? Yeah, so that would be in a transcript. And just to repeat, we want to run the same experiment on pre-trained.  
Essentially, because you've been running it on HPC, so can you generate embeddings, 760 size embeddings for the test set?  
Using the quen pre-trained model, is it? I I did some research before this meeting, and so there's a quen 3 and then dash embedding. Is that like the embedding model or is it like a general mix of?  
But this.  
So, that.  
There's several types.  
And if you search for embedding, there's going to be a specific embedding model, so...  
No, but we were looking.  
Tomorrow, they will be.  
No.  
Mm.  
Thanks.  
So, we want to know what model we're using for...  
Which set of experiments? The one with training or the one? Yes, this is the one.  
Why I didn't want to use specific embedding models is that we want to use general models that can do everything, and then we pick that, and then we ask it to do one thing, and then thinking, because if I if I pick a specific kind of model.  
The initial test, when it had no objective, should be gender, right? What do you mean? You're supposed to do everything right? So, what do you think?  
So, this is the model that we've used right now for training.  
And I wanted to look at how this model gave us the embeddings before training Azure, that's why I asked.  
Daniel to June.  
So I'm going to try to run the experiments now, set them up.  
Me.  
Ohh, also another thing changed: go ahead, this get into the number of points for the video distance.  
Number of points. Does that make spikes that happiness or love? Yes, that is.  
This happiness, happiness, this is happiness, this is love, but essentially all of this, if I'm just looking at pattern, then essentially they were all like this. They were kind of matching very closely. Now there's a separation. So there's a high intense or like separation and.  
There's A offset offset. It has six closer percent right now. What are the two? What are the two that have the highest? Love and happiness. Love and happiness. And that's probably because as humans, we have we have a more sexual point of understanding what that means compared to like fear, which is like a massive feeling.  
Can be true, but I mean, definitely that can be true, but we can conclude that only once we know that the data set was balanced. Yes, yeah. Also, it could just be that there were more data points for happening. So, that's an unbalanced data set. Unbalanced. Okay, I better push all the results. No, no, I don't know.  
That's what I'm trying to, so if you look at the name of the folder on the top level, it should say TMPQ and then B or U, and if it says B, so it's balanced, balanced, yeah, so you're right, it's balanced, it is balanced, that means that...  
If love and happiness reach more closer to centroid, then that means they are closer to being pure emotions. Well, there's no such thing as a pure emotion. It's just our human definition has become more of a consensus on these two. You see how it starts at 20, so there's nothing from you.  
Wow, and in the in the previous experiments, they started at 7. This is a different model. Oh, it's a different model. So it doesn't count. That's why I ask you that can we generate the embeddings from pre-trained coin models. Right. So we want to generate.  
Uhh...  
So, when we...  
So, so let's work on that.  
What we're trying to do is, so you have two sets of experiments to start with, two different models, and then it showed the patterns are similar. Yes. So now we're looking at this pattern, it's different. Yes. And then we're saying, well, actually, we want to use a different model here. Is that what we're saying? No, so what happened is we initial test without any objective. We did it.  
Let's say model one, model 2, data set one, data set two; we found out that all those combinations still bring in the same output.  
We trained Model Three, right? This is what we're looking at here. Gwen is Model Three, right? This is trained Model Three. What I just want is to just look at untrained Model Three and see that most probably it will remain in the same pattern as Model Model Two, but just for just for assurance, just so that we're, we're, you know, we just...  
Just to confirm, you know, this untrained model 3 doesn't have this before training one. Right. So, it probably won't. Model 3 shouldn't have what we had in model one and two. But just to confirm, just to know that. And for models, for the experiments one and two, it wasn't a square 1.7 billion.  
This model we're looking at now, it was a different model. For experiments one and two, it wasn't Quen 1.7 billion. Yes, it was. Okay. We're looking at smaller models because I used to run them on locally on my system. Right. I was trying to use clip. It's a lot smaller model than Quen.  
And my MacBook heated up. I had to, wow, listen, like, wet my hand and then rub it, so then it's wow, this doesn't have a fan either.  
Okay. Have you used just one singular source of data for all of this? You used one data? Two data sets. Actually, 3, one binary and two formal classes.  
Okay, so with these two for multiple classes, this is one of the data sets, right? Yes. Is this pattern reproducible in the second data set? This specific pattern, the second, the train model? The train model will happen. It should be, because in the briefing we saw that  
Data set one, data set two, model one, model two, everything gives out the similar pattern, yeah, so we just assume that if we, I don't think if we need to repeat that thing for every experiment further on, because you said the ground rule that, irrespective of model or data set, the patterns could be, so this is this is the only thing is the.  
The only reason I've said that is because there's something that's changed in this situation, which is the fact that you've trained it, if that makes sense. That's why I'm saying that however I want to look at model 3 untrained, which should probably be something like this, which we saw earlier, but just to just to just to just to make sure that model 3 didn't initially.  
Even pre-paid model 3 bins for this, which is good, but you know, just to just to be sure, OK.  
But, from here on, I think we won't have to check everything for every model, because if we we said we have set ground rules that, you know, at ground level we saw that patterns remain same irrespective of model and data set, so we move on from there if we keep on repeating this, this, this, this, this time, yeah.  
Right, and again, the only reason I ask is because this is model 3. This is model 3, but also the pattern. The pattern is generally the same, but there is the difference of seeing as this is balanced, there's like a separation. There's a line there that you can separate how these two behave.  
Yes, those who behave similarly, those who behave similarly in the initial experiment, all of them. So, yes, that's a new thing, and that thing, and I think I think that the language of your thought is saying that some pure love and happiness are getting closer to being pure emotions.  
than the other ones. Yeah. You can be purely in love. I can be purely happy. You know, I'm just happy without no reason. I love you without any reason. I'm feared because of something. I'm angry because of something. There are levels that kind of like what we define in emotion. What are our own embeddings?  
This is new finding, by the way, that this, yeah, behave differently from our motion, because up until now, every motion we had the same way.  
I would like to see this work on an embedding model. I know I'm still stuck on the idea of choosing a model, but I feel like the general models are no mixture of experts, right? They're trained to pass all the benchmarks, so they have an expertise on coding, math, and...  
That stuff today, and I brought you always today, so that by the end of it, I can, we can all have tasks, and if that is something that you're interested in, then by all means, you, you can have, you can take a task around that, so that you get to explore that, and we'll all explore something we need.  
So, obviously, the submission for this. So, at this present moment, what is the structure of your findings? Do you know what I mean? Like, what do you aim to, what is the aim to?  
Put in the paper, and what is like, what is the kind of story here? You know, this, this what we did until now with X after training, the one thing that's left is we had to confirm with the pre-trained model that the embeddings before pre-trained, yes, if that is that.  
Then all of this, that experiment replicated for image.  
Everything we did until now for text, we implicated for image. And then we say that regardless of input type, text, image, video, when video, I don't know if we can finish it. I don't think we can finish it. So the structure is we thought this is what we were looking, well, we found this in a book like using text and emotions.  
We found this in images. These 2 things confirm that this is how the model, regardless of your input, this is how the model shows context.  
In our case, emotional context, you know, you have to start somewhere. You have to take one second. Well, that is perfect because it is it is also just mapped for people. You know what I mean? It's not for people. And it makes sense. And you know, we still realize certain things. There's a lot of happiness thing. This in itself is a huge finding.  
We have mathematical proofs that love and happiness up your emotions, that hate, anger, and fear.  
Two.  
That's it. And it's like, so from an explainability perspective, people are saying, well, all these models are black box. And we say, well, actually, there's some things that can be explained here. For instance, that happiness and whatever, they kind of, they're over here in a finger.  
And if you gave a model a phrase blindly and said, there's the model running on its own, I'm going to bed now. If anyone posts any abuse here, make sure you flag it. And then, well, guess what? It falls, hopefully, it falls in that cluster that will be abusive.  
close to anger and stuff. And you say, oh, sorry, we're going to have to send this to a moderator because it's too close to a cluster. Oh, OK, that's really, yeah. So it's kind of explainability that. Before the model says something, just by the part of the model, you're able to say, you know what my problem is as a person is that  
And it's, I'm so different to you, is that you, you think...  
you think ground up and work your way up. And I almost always try to navigate from top down and it's not on purpose, do you know what I mean? So that has given me, that has filled in all of the extra stuff in my brain and that the whole thing makes sense to me. Does that make sense? Yeah, so there's the set of experiments that I wanted to track.  
But can I tell one more thing to Josh, right? So the first hackathon, we were waiting for the pizza downstairs. And he says, well, this is my idea, right? I want to have this neural marketing thing where you know by the video if it's more likely they will sell product A or product B. How do we map this in this somehow do it? And then this discussion started.  
So, it's the idea came from the top down. Yes, you know, I know that now, but you veered, so we talked like a week ago, and you had veered so much far off it that I was like, "Oh, this is a new problem," and I've been this whole time, I've been trying to, like, I understand the implications of it, I understand what it can be useful, but I didn't understand the general problem statement. This makes a cluster of products you want to buy.  
It is a new product, it lies out or you will buy it. Well, now my thought process is, well, you can define the clusters, you know what the clusters, so if something falls in the cluster, then you know. Whereas before I was like, well, I know we can define the clusters, that's very interesting. But what does that mean? Do you know, I mean, now if it falls within this space, you know what it means, because you have...  
kind of an understanding of all the possible clusters that could exist within this dimensional space. That's the black box, trying to solve the black box problem there. But in the part of the model we're trying, we can say that. This is also kind of like how brain works. I was saying you're right, when I say that, that example, when I say I am sad.  
You look at multiple bubbles there, you know, you look at the thing, oh, Pritish said this is what he's going through, or there's a train strike today, or there's that, your thought goes through multiple context bubbles blamed by you. Like how, in the same way that you're training this, this extra model for emotion, it's the same way that  
If you tell me something that is, and I already have the context that I, as a human, I have the context that we're talking about something emotional, you say, oh, I'm feeling sad. The part of my brain that activates is probably, yes, because of the hippocampus. And so what you're doing is you'll find  
Well, this general geometry in this LLM's millions of parameters, this geometry over here is the emotion processing kind of area. Yeah, so very cool. Well, that's an interesting one in itself, because...  
It's like, if this thing is a brain, you start to figure out, you were saying last week, like, this is like, it's almost like, well, this is like, you know what I mean, because the parameters are tiny, but you know, humans have...  
But because then all sorts of crazy things with vision can be done, right? Because I imagine there's a huge body of theory of how the kind of the visual cortex works and the nerves and the images. I mean, the whole idea is that if neural network is based off of human neural network, then it should behave like human.  
And that in itself is something that's hotly debated, right? Some people say, no, it's got nothing to do with it. It's a model, but I mean, I don't, I think it's got something to do with it. I think there has to be something. I mean, if you're self-aware and you know how your brain is thinking of something, then you can use that to try to think, try to...  
Try to predict how the models predict. Okay, I have a question for you. So, I'm prompting here to say, right, we use Quinn for the training. What other models were used? And it says, well, an all mental M6LV2, an E2 base V2, a GTE base. It's coming up with those three.  
I don't know if that's hallucinator, so sentence transformer, it says it's using that model in floats, it's using that one, then help, it's using that one.  
I don't think you got it right. No. OK. We've used these two models. Yes, we've used these two models initially before. Yes. So what we want to do is say, get the quen model, repeat the experiments with these two, and generate all the data that you generate. No, what we want is to take retained quen 3  
Point 5, yeah, 1.7 billion parameter model, yeah.  
Give out the last hidden state 768 embeddings.  
Using the text. Right, so we want to say, repeat this experiment here, but instead of using this model, use Gwen.  
Repeat the experiment where you use this model, but use Quen and stash it in a directory somewhere. Got it.  
Crisis of birth.  
The idea is that we read these experiments, text will be concluded once we have the pre-trained. Have you got a preference to do the BG base CNV 1.5 with the other ONP net base V2? Any preferences or the first one, the BG base?  
Yeah, no, I've already used that one. We just want, we just want Quen. I know, but basically what I'm saying is I'm going to tell the model we want you to drop Quens and repeat what you did with this one with Quen, but they're going to be the same thing, right? OK, cool.  
We repeated the experiment for NPNet one, so it's the same thing.  
Yes, so now to repeat the the same set of experiments of images is where you guys are coming, because in images, well there is 111 possibility that we repeat the whole thing for emotions. I already have that data set.  
I, I, I think I'll conclude that soon, but we want to do other set of experiments as well, that's right.  
So now to the context, what you were talking about, that right now you only covered emotions, but what if we take other type of categories where one thing can be absolutely one category? A dog is just a dog instead of a mixture of other animals.  
But for emotions, I have this data set that...  
I have the pre-trained embeddings on, so it's essentially its faces, its faces of...  
These is angry faces.  
Right, ohh, genius. It's like, yeah, sad faces.  
People are making like apps with camera that detect motion. Yes, this is just this is classification, so, and that's not what we care about. We don't care about how accurately can a model predict that this is sad, because there are models that are 100% predict that this is sad.  
What we care about, this is what the model said, this is what the model thought. You know, the idea of said and thought, we want to find correlation between what is being said by the model and what is being thought by. We want to know how it gets us to. Yes, we want to know how it gets you. We want to repeat these experiments with this. Yes, so one experiment is to repeat everything that we did, which is...  
Uh, and create embeddings, plot those embeddings, give it the model as an objective, train that embedding and plot those embeddings, see what changed.  
Yeah, just send it.  
Another thing, but this is for emotions. Now we change the set of categories. We take dog, cat, animals, which can be absolute. Now these categories, I hope, can look at the image of a cat, it is only a cat. So now we, now this, I am imagining that even when the model does not have an object.  
somewhere in somewhere in. Yes, it should be a lot closer to dogs. But in that particular thing, we have to make sure now when I look why I picked this data set.  
There is no noise, by that I mean that when you look at this image, it's only the face, it's only the face, and the emotions, it's not a thing, but when you're looking at a dog, imagine there's a dog and there's a tree, there's a sky, we don't want that, you want to, so that is where the control context scheme comes in.  
By that, I mean that gift.  
Let's say in this image of dog, we use segmentation, you know, background Google.  
We remove the background, so we remove the background with segmentation. Segmentation is a method to do that in background removal. I'm sure you guys know Apple does it. If I click on this image and drag, then only the dog comes up. That's what I want. We want to take only image of the dog.  
Create embeddings, look at what happens, and then maybe one with the whole image, create embeddings, and look at what happens. I'm imagining that in this image, because when the model can bring this image, it has to keep the context of grass, has to keep context of this something back there, a multiple other things.  
But when we take out just the dog and put it in a white background, then we imagine in this context, it will be almost similar to the emotions in that density will probably decrease as you go farther from seeing the voice dog. Well, that will be a lot closer to the centroid, because  
Emotions cannot be absolute, this is a fact, this is absolute, this is a doubt, but there will be, there will be space around it, because you've got loads of different looking girls, loads of different, whatever, but there will be a central point, or there will be an almost central point, even if it's not like, even if it's not directly on the central.  
Like, yes, even if there's no point of density, there will there will be a much smaller ring around it. Yes, the smaller ring, yes, definition for it, more just regards, so to set of expect images.  
His emotions.  
No.  
Absolutely.  
These animals.  
Yeah.  
It's absolutely not enough of that. There is a third one that I don't know if we can do it or not, but that is the one that so in emotions you saw how surprised it was, it wasn't an emotion, it was a reaction. So it then it remain kind of.  
Out of the restaurants, but still there. So, one experiment that I wanted to track was, I don't know, maybe you guys can help me come to the category, but it's a few absolute categories that are distinguishable, and then one while you come. Let's see how they behave. So, in my how I thought would be that...  
Dog, cat, running. Imagine if a dog is running, it's something to be dog is running, yes, but but dog animals and actually running, I don't even know, so you said animals are absolute, there are multiple different types of dog.  
Do you know what I mean? So there's a general in this final experiment, are you really trying to control the absolute point? Because, in that case, by absolute, by the way, I mean if the glasses are dull,  
There are more pens there, by the way. You know, just throw that one away, so you know it's dead. If you put it under the desk, I won't take it back. belongs to when you're talking about sad, happy, and love. It belongs to both those classes. But in this one, if something is dark, then for sure.  
And then, so you do something like dog cat, and when you say running, it's not necessarily a dog or cat running, it's just running something running. Yeah, that's the confusing bit. It's difficult to have a same image of something, everything, how do we create a centroid of running?  
Using images? That's a video idea, if anything.  
Yeah, again, I mean, this, this, this is what I thought that maybe could be done, but you know, open to suggestions, something that is something that a list of categories that are absolute, and then one wild card entry that is just remotely close enough, so like emotions with surprise, surprise was a reaction.  
Yeah.  
Oh, by the way, also one nice note thing that I noticed in this surprise is no more a wild card entry, because now when we look at faces, all of the faces have reactions. We're no more talking about sad, we're talking about my face being sad.  
Yeah, my face being surprised. Now, surprise is Gomora while currently, so let's see how it behaves, because now all of them are reactions. It's not more emotions. It's all of them are reactions to the emotions. Interesting.  
I was gonna say something, but I don't think it's appropriate in this. Go on, fire away. Man, woman, baby, wildcards.  
Man, woman, baby, wild card. Could be, could be man, woman, and or underage, underage baby. Not the worst idea, but when we publish it, it might be against the LGBT community.  
Plus, yeah, pay more contributions.  
Hey, don't say that in a way; we cannot mention those things in the future, and then L.L.M. assume.  
No, I think. Wow, man, that'd be interesting. But that's the thing that my PhD thesis concludes. It's like, there's no point in trying to eliminate bias, because bias is an intrinsic property of the system. The thing is quantifying and understanding where the bias is.  
So if there's like some LLM trained on LGBQ data, it's like it might have 100 genders, and then one that's trained on non-LGBQ might have two genders and that kind of thing. And showing that the biases are intrinsic of the system, and then you can try cleaning the system and deleting data and whitewashing stuff.  
But anyway, that's kind of that's difficult because the world.  
Will have that kind of, you know, noise or biases, or whatever you want, always be, always be biases, that's just something to be employees for hosting, yeah, it's trying on this, trying on.  
People are infinitely complex. It's a lot.  
But we no want to judge how the world is, we just want to do.  
We just want to do it now. That's A t-shirt, right? Guys, we just want to do it now. You know, that's our characterized coming through how the system works.  
Yes, so these are the three set of things that I thought for, if there is some of the solutions.  
The running is something that can confuse your your the reason you put running in this golf.  
Cat example is because it has a relationship to both, but it can be confused. I think, I think I talk of because of videos, right, because you are also going to do videos, but now for the time communications, we just want to finish images, but I also wanted to do multiple category sets of images, because emotions is something that can be, you know.  
It can be a multi-label thing, but we have to have categories that are not multi-label. That means that they're absolute.  
Creating the category works.  
A...  
Would imply that you already have the data set, otherwise you would have to, no, I mean, if, if I mean, think of a category that we can easily found the data set, so, so, ideally, we look at the dog and cat animals, we can easily find images, so, ideally, we would look at the data set and then see.  
Like, detect a category ourselves that we find among the images. I mean, I thought of the category motions, and then I looked for the data set. I mean, models.  
car, plane. Yeah, but yeah, it's easy to confuse occasionally, but you usually know what car, plane, boat is, you know? Yes. I'm sure there's a data set. I was like, that's what I'm doing for my new computer. So imagine that there's a model that was trained.  
on that kind of vehicles. And that's the VLM, the vehicle VLM. So, and then you give it things, animals that have nothing to do with vehicles, will it cluster it somehow? I think there's sort of, I mean, this could go on to three generations of experiments down the line, Europe's 2030 kind of thing.  
The only reason I'm saying vehicles is cause that problem, just because of, you know, vision, there's loads of like, ohh yeah, and also very vehicles, black vehicles.  
Yes, does that make sense? Cars, trucks, airplanes, black vehicles. So somewhere if a car is black, duck. If a truck is black, duck. If an airplane is black, yeah. Yeah, that makes sense. Interesting.  
And then there's the reverse thing of saying, let me know if any black vehicles drove through here and then if the embedding, it would be something in those lines. But how do we get the black cluster? But I imagine that would be a case of.  
getting a set of images and saying, where are we here with these things? And then you have the vehicles and you have black, you have the vehicles and you have red. Where is the point being pushed to? And you figure. Oh, yeah, yeah, right. I mean, I think so. But we are assuming that the direction of those points will only depend on colors.  
It would be multiple things, right? Well, it would be like the emotions themselves, right? They spread, they like some go to surprise. I mean, surprise falls in the middle. But if these are hypothesis that they're questions we're asking, right? It could be that can be determined or, but I guess the intuition is that somehow if the model was trained,  
that somehow it would capture colour and be able to say that was a red vehicle, that was a black one, if there was enough data. Yes, I mean, you're actually right, but how we had to form a cluster of surprise, we needed data points that were labeled only surprise. No, we did it.  
OK, you did the final, the final label had to be that there was a bunch of contextual information before, which came from the attention and the weights and all that stuff, right? Yeah. In the same way, if you have a picture of a car or whatever, you're going to have in a data set a million things you're going to have, like...  
What colour was it? Was it a car plane or thing? But it's going to have loads of different things.  
Those are still all going into the model. Do you know what I mean? Would you not put all of those things into the model? It would be, sorry, it would be training for all of those. Yes, it would be training, yes. But without our training, I'm saying the first set of explanations, and we look at no objective given as an image. What do you think? Well, you just cut down it.  
a big data set that already has picture and the details and you take everything out but the color. Because guaranteed for like a speed camera system on the road, well you know these things that flash and then take the data and work out, it'll have like license plate registration, but it will also have color of the car, model of the car, you just get rid of it.  
That, and you only put in.  
You can encode, like, on hot encode black, you take all the colours. No, I'm saying we just, so the way we get embedding is we put an image, we get an embedding. We, we, we cannot put, we cannot put the context specifically with the embedding. I'm saying you've got a data set, right? Imagine like a traffic data set.  
Which has, I get what you mean, yeah, and we can have, so we can have the labels car red, car black, yeah, plain black, something like that. What I'm trying to say is that, let's say there's a cluster here around car for car, there's a cluster here for...  
Bus, yeah, I don't know, bus, the cluster have for something truck, yeah.  
How do we know that? How do we get the black just the black cluster?  
How do we get the cluster? Because if the car is black, let's say there's a point there. Yeah. The truck is black, there's a point there. But for us to say that, you know, they're all pointing towards one point, we have to find that one point. That's what I'm saying. So the data set, you have to.  
You have cars, you have the colour, whatever. You take the data set out. Now you've just got cars and colour. You take the colour out and you encode it so it's zero if it's not black, one if it's black. And it will, if what you're saying is true, black will naturally form a pattern anywhere. Do you know what I mean? Does that make sense?  
Yeah, that somehow the model would capture color. Yeah, I was wondering if this should have been maybe like a like a two-phase.  
Uh, experiment like one would be for models, or the other one would be for color.  
Models of of the car, yeah, yeah. You assume, you assume the most basic thing in a vehicle recognition software would be able to do one of the most basic things is take colour, like the colour of the car, so that colour will always exist in the data set, right? But going from just taking out black, just saying it's either black or not black, you have the labels for those.  
If what you're saying, your whole theory is true, that black should form a cluster anyway, regardless of, regardless of.  
Do you see what I mean? Do you understand what I'm saying? Oh, got it. Yes. So, so the data set has...  
The label.  
So it's just the label is like red, blue, and you get rid of all of the red, it's just black and not black, that's it. Got it. Does that make sense? Yes, that is the start, the start adding. So what you're saying is that let's say we take the 768 embeddings, correct me from wrong, okay. The 768 embeddings represents the image that's got it.  
Okay, wait, we plot it. At once we have car is circles, and let's say bus is this. Okay, we do this. And then same embeddings, what we do is, okay, now do one thing for me, this black and non-black, regardless of what vehicle it is.  
Yeah, then we start seeing the blacks.  
Yeah, blacks and not blacks. Yeah, then we try to look, then we try to look at them together, yeah, and then we say black cards, yeah, yes.  
And then, so there's there's what you're doing is the only difference is is that you're clustering for two things separately, but you'll be able to, because it's going through the same, you'll be able to see, yes, that that makes some sense then. So, yes, because the end and in this case it has to be.  
That the embeddings we use are same, so it has to be the same embeddings that is at once clusters for type of vehicles, and the same embeddings that is at one, what the data point is.  
When you come.  
Yeah, right, you're right, you're right, you're right. So, once if we pick up build 768 building at one point of the model, then the same way that you use like all of these are angle by the centroid, it's like all of these are black by the centroid, all of these are wi-fi, all of these are cars by the centroid, and if there's two bands here, let's say the cluster.  
There's one cluster for it that's like that for black and non-black, and then say you set that like 5 for opacity, and then you have clusters for cars.  
So, the browser of that statement now becomes.  
Automobiles.  
Will wildcard be cut?  
Please, have a explainer.  
That's.  
Yeah, certainly. There's some type of category that is not absolute, a type of category that is absolute, and the third, we're trying to look at multiple contexts from the absolute values for all the categories, but there are more than one category. So, separation, yeah.  
Yeah.  
I mean, they're still absolute with a different kind of chemistry.  
You're a man, you're also a human. Yes. You're different from a dog and you're different from a female, but in different ways. Yeah.  
I have to look at the repository.  
Yeah. Yeah, should we have them? You can just give a general structure if you wipe out. Well, I suggest you get Gemini and say Gemini explained that thing to me. I'm trying to keep some. I mean, I don't understand the whole thing.  
I'm like, "Base, remember what I said about top down, yeah, and now we know why I feel like my brain's fried after a long meeting, and I tell that to Daniel after finishing the meeting, I think my brain's fried right now, yeah."  
So, this is what the repository looks like. Sorry, I applied.  
Yeah.  
These meetings and minutes of all of the discussions we've had, Daniel updates is every time we finish the meeting. So the last update is, the last update I actually didn't put in there because it was a mixed thing. But it's, so that one I owe, but that one I think was like, we need to get together and discuss how this image thing is going to work. So  
I mean, it is impressed.  
This is also 16th. Yeah, 10th and 16th is there.  
And, and then this prompts, so whenever we try to.  
That these prompts will have when we try to, you know, what this is similar to what we've learned, yeah, yeah, yeah, there's only there's only six prompts so far that we've done, so it'll be very quite big leads between these things. The thing is that, after one point, I started putting two prompts, right? This is, this is, this is...  
Yeah, so is that the vid IQ thing? Do you want to pull? Because I just pushed a bunch of experiments up there. Do you want to do a good pull on that thing?  
I'm assuming, though, that all your experiments have like a nice little read me or something at the top, or something that tells you what it is, right? Yes, they do. Yes. Okay. Because if you know what's there, that's fine.  
I think I've generated some things that I would have to push to. No worries. Yeah. I'll do that later. Okay. Because if I pull it, then it's this.  
because I also made some changes. I'd have to stash them and then pull yours, then put back mine and then push mine. Yeah.  
Anyways, but I can look at the experiments that you put on that. Well, basically, the one that it just finished writing is the one we just discussed, where it's going to drop the Quinn model and say, regenerate those embeddings for us. Oh, did you do that? The next one? It's in the, it should be on the, so there's the commit. There should be a commit of a few minutes ago.  
Two minutes ago, is that it?  
It's and it's a batch file, so it's a dot slurm file.  
Yes, this one. Yeah, that should be the one.  
Oh, but we also need the empty Wi-Fi, right?  
The embeddings will be the ones it's going to generate. Yes. Yeah, but that's the expectation. We run the experiment and then it's going to generate the NPY files and I push it back to that repo. Okay. I haven't run the experiment yet. I'm lining it up now, so hopefully it runs. Yes. You don't have to train anything, just...  
Load the pre-trained model, generate the embeddings from one, and...  
Right.  
So, there was a...  
So, how I've been doing it is that reports has all the reports, so you know, when we look at image data set, what do we look at with text data sets, what do we look at?  
Scripts just do you take the embeddings, so you put the pass stuff through the model.  
You take the embeddings, you move those into a notebook, everything is good.  
Now, what you we are going to do is inside this experiment, because I think that's the way to operate from there, and now we will all be working together inside this experiment, I made text embedding fields and text model, so and in images.  
In images have created emotions.  
Sub-folder.  
I'll suckle this for up.  
Yes, so inside.  
Yes.  
Yeah.  
And experiment.  
This is the repository in experiments. There's text, text model, and embedding speed. Embedding speed text was when I initially just looked at the text embedding feed. Now there's text inside the text, there's binary and multi-class inside images are created for emotions.  
which is this one. We have another folder inside this for this one, and another folder inside this for this one. Now inside these folders, I usually have reports that will have what I've done. So for image and report, emotions, image what I've done is there in the report. SRC has the scripts.  
Yeah.  
And this, this is the data set that I use inside that have data. I mean, essentially, you wouldn't have to worry so much. You just make a folder inside images for a task, and then you just write code. Because in itself, when you're working inside that folder,  
Gemini or Codex, whatever you use with itself, you've got three different ones so far that created three different sets of environments, correct? Yes, so there's only three sets of environments, yeah, and do you know we could have a Vibe coding session during the week, like a one hour.  
Yes, I think it's, you know, let's get together and say this is what we're doing. We can look, all of us can look at what the other person is doing, and that way I think, yeah, if I need to finish this course. And have you guys got HPC accounts? I just got approved.  
Super. All right, cool. Perfect. So that means, I haven't set up. If we have, if we have 4 accounts, that means that we can run, we can absolutely burn those GPUs. It's all the separate. I think you two have set it up and actually used it. I have the access, but I haven't used it.  
I think, see, is it running Linux? Come on, yes, yes, the way you set up, no, it's OK. That's fine. I'm going to we setting up isn't that much, it's more like how you prepare your projects to run up to, yeah, that's why, that's why.  
And yeah, just, I can send you a few commands to set up that. I think, yeah, you had sent it earlier in the week. I was saying this in the British earlier in the week when we were doing this coursework, you sent it and you were like, oh, it took me a few hours to set up and then I was good. I was like, at the time I was like stressed out with this work.  
So, I can't afford a few hours; I'm just going to do the two models took two hours anyway each, so I just, and then after that, I thought I was being stubborn because I was like, "Well, I really didn't create the account, so I just continued today. Let me see if this model is going to come." What I realised in this project so far, we've done MLA, we've done some mathematics.  
We have test physics with the concept of density. We have test spirituality without being to think and everything and the emotions are there's no pure emotion and all that.  
And I think with images, we might touch one more subject than biology, because now that we look at animals, maybe we found that, you know, herbivore, no, I think in images it won't be about herbivore and carnivore, in images it might be about that animals that have...  
four feet, you know, walk on four feet, act differently, animals that, you know, the way they look, they form clusters closer if they look similar. So dog and cat might be closer than something like chimpanzee or elephant.  
Yeah, then we get biology.  
Okay.  
OK, but the restrictive matters as well, which sites the animal can be harder to see.  
Okay.  
See.  
It's getting late. Yeah, I think it's home time now.  
It was just with not time for us to explain what we have done so far. Well, I mean, but that I think puts everyone, it's like it's up to speed, so.  
Burns of passion.  
Yes.  
Looking like for the video, yeah, it would be similar to image of just being under the coral space, like for each second or each half second, would map map it out.  
Say that again, you were thinking about the videos? Yes, so the video would work similarly to the image one, but it would just snap it up for each timestamp.  
We will divide the video on segments and then switch up those and then use that to embed. Yeah, and then have, well, I imagine it would be that sequences, it would have to be frame by frame.  
And by frame would be maybe too little of a difference. OK.  
Every half second.  
Oh, I'm excited now. This is so close to running and it's crazy because he mentioned this.  
So Pritish mentioned this maybe 10 minutes ago, and then it's like, okay, so now this thing is going to actually run on the HPC, seriously.  
Submitted job.  
So, it's, yeah, it's still haven't installed it. Scratch, which pip?  
Aspatch run Quinn Python.  
Yeah, I don't think it worked. I suppose you were able to run the Quinn.  
Model on the HPC. Well, it's got like loads of memory and it says 7, it's under 10 billion, so it should run a 40 billion 40 gigs. I was thinking of like the way of getting it onto the.  
Is it Olama or you just?  
I think I'm using Hugging Face. Hugging Face, okay.  
Reinstall.  
Okay.  
Bye.  
OK, but there will be one difference in four minutes, so, you guys go to somewhere after.  
He has to work on your computer. I, I done.  
I've done all of the models; I haven't plotted my graphs completely. No, you probably use a notebook for that, and then work on it on the results.  
You do EDA on the data set, and then you do the EDA on the results as well. Come, let me show you something. I just want to, because you've done that, so I just want to know. I need to, I need to transformers in CNN. Literally, I'm worried I've not done.  
What they've asked, basically, for all I've done.  
I did a baseline model based on my original stuff, so I very clearly set.  
Then, based on that, I did a random parameter search on, like, things that I thought were defensible. So all of these parameters are defensible from literature. Does that make sense? Yeah. But you're still supposed to do the grid search, and then we say that, oh, that equals right.  
See, this is what I was saying. I did, I did a random search because after like a while of researching, it was, I was kind of, I get to the conclusion that if I do grid search for like a lot of different things, I want to have like the kind of, I wanted to find a space that was in place. I varied everything. Did you also vary everything?  
Run, Sir, and Raja.  
Mobility doesn't matter if you're using Adam. That's S to me. And then you vary also the hidden as well. So I did random search for one hidden layer and I got.  
Right, yeah, results.  
I can show you some of it. And then, so I was like, well, two layers. So in literature, I found that in the literature, you would have fun. So it would be like bigger size, smaller size. Yeah. So in my random search for two hidden layers, I did flat.  
architecture, funnel, bigger funnel, inverse funnel, right? I run that and I got my output, which is like, has improved by such a tiny amount, but it's 0.3931 now, and it concludes that, okay, well, the funnel architecture is good.  
a smaller funnel is better than a bigger funnel. It's actually dropped the epochs here. The learning rate is actually pretty similar to the original. The batch size has changed a bit, and the weight decay is smaller than the original baseline model, but the one related model. And it's supposed to write a bigger model. Yeah, which is supported by, which is the like some of it is supported by different pages, but it's supported by.  
Again, I don't know, so SDM, I just want to run, I, I did different problems, nothing, I don't know, that was a bit of a walk.  
I did, I did, see, the thing is, I think.  
So, I did spend a lot of time asking all the questions, and by the end of it, it was like, you really only need a very few of these, you use a very obviously gamma degree, and then I can't remember the sequence, but anyway, I've got mine here and it was 1.932.  
Right? That's where I left it. And then I was going to look at all of the results and be like, well, these are all...  
Fairly similar in classification for the slide.  
like SVM was slightly better. I will have to show you what...  
That's good.  
Did you put out SPM?  
There's the, there's a thing running on the HPC complete finish point parity variance entered out too. So it's generating the results and then hopefully we'll have them and compare the thing.  
Sorry, say that again. No, I forgot. I forgot it was recording. Is it going to take any of the context of the stuff up and say it's completely not ready. Oh, it's all right. We should be able to say and then coursework was discussed and. Yeah, ignore the coursework. I started talking about probably about two minutes.  
Yeah, even if you say that, it doesn't have any concept of that. Yeah, that's it. It's transcribed there, Josh. Sorry, made that blah. So it's all there.  
Even if you said love.  
Complete. OK, so is that it? Is is that all all the results we need? Yes, there will be raw L2 and centered L2, but we just want to look at raw. So, we can, because you repeat, but later on, you remember we talked about how raw is the one that we have.  
So, it finished. Oh my God, that's shocking. So, I can push them and we'll have those plots. Yes, that is. If you push them, then I have to.  
I have to fix my response.  
Just to be able to push it, but I have to fix my. But can you not tell the agent, merge the stuff? That's what I tell it and say, work out these things.  
Oh, no, I can't because I've been used up my codex again. I do one experiment and I'm caught with the codex used and another pro of it, right?  
And you still remember.  
Have you been using codecs? Yes.  
But you know what, from an experiment, it doesn't even matter what agent you use, because it's scientific process. Yes, the scientific process is all about. Yes, it's not about.  
OK, it's it makes sense.  
What?  
Right.  
Oh.  
Bing.  
Right, it's... Did it generate the plots? It didn't generate the plots, only the...  
The JSON files, I have to ask it to generate the plots.  
No, no, you can give it to me. I'll tell you the course. So they're on...  
Yeah, so in multi-class, the AI emotion runs, there's a run 303.  
And then there's run 303 Quen and all the stuff is there.  
Yeah.  
Pakistan, Pakistan.  
The.  
So, you use that data, and then you run your pipeline, and then you generate the plots of that stuff.  
you run your pipeline and off those JSON files and that it hasn't generated any NPIY files.  
Confusion matrix or Jason summary dot Jason progress run method. I don't know. Well, we'll have to see if that is adequate. Did you push? Yeah, because that's what it pushed over there on the screen. And it's saying that there's no NPY file there. You asked about the NPY files.  
So it looks like it didn't push any.  
Embeddings.  
That's strange.  
Yeah.  
You see, you generated this for the...  
Inside the tree, the one push that you did right, so I just, I just kind of, this is exactly what it looks like.  
There, I have motion.  
Things.  
Models.  
Three.  
I would include only two.  
Yeah, I know, so we need the embedding files there.  
Ohh, \*\*\*\*, you talked about speed as well, uh, sorry.  
And we have a question.  
I don't think speed is that relevant when you keep in mind the architecture of the. It also pushed some results for the.  
The training, like 50 epochs, 100, 250, I don't know if that's going to be useful, but I imagine it would show a similar pattern of like 2 humps since. But we don't, is that more relevant? Right, is this experiment here with the Quen model?  
Uh...  
It is even strict in my Shogun's name.  
Okay.  
Are these all of your grip satches that you've run with?  
Sup.  
Yes, exactly.  
Two.  
You didn't see that.  
Ohh, yeah, just pre-training is not allowed to do this.  
Which we did in part one, so we really, yeah, it's just my my problem is so much less complex. Ohh, I think I know what's happening. I'll have to simplify this one, OK.  
Right, okay, I can see what's happening here. I'm asking it to ignore the NPY files. We want those NPY files. No, because like with the other, with these models, the ones that I was...  
that we did with a queen.  
They're enormous files, so like gigabytes in size. Yeah, so as an output. As an output. So, and then I said, well, we're not going to include them, but I'm going to ask it to make an exception and force these NPY files, and because we need to see them. Gotcha.  
Stop.  
Alright, got it. We do have the stuff.  
So, British, we do have those files. It's just that I have to force them in and then we'll have the embeddings.  
So...  
What?  
What's going on?  
OK.  
Ohh.  
Yeah.  
No.  
Copy.  
Ohh, what is this?  
Yeah.  
Good, good move.  
Play.  
Okay.  
Call.  
Play.  
Call Aridhi.  
Yes.  
Makes sense, so it's pushing, it's pushed, so those NPY files should be there now.  
There is a, so and lots.  
Sorry, say it again. I thought I heard the little plots were both. I heard there was a problem.  
Maybe because it can cooperate. So, so you're done with these guys, and this is the...  
I'm sorry.  
Although, I think that one seems up for the last stage.  
Yeah.  
You have a nice.  
I had a, I had one meal, I was quite, I was thinking, I'll be careful.  
Really? Yeah, my mom went to... Wow.  
And then, when I texted you, ohh, she's back, she's back, yeah.  
Okay.  
I don't know if you can achieve it, but...  
I have to finish this thing.

Sikar, Daniel** stopped transcription
