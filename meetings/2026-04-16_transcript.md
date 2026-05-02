**New Dataset, same and new model discussion-20260416_171712-Meeting Transcript**

16 April 2026, 4:17pm

1h 12m 54s

Sikar, Daniel** started transcription

Sikar, Daniel** 0:30  
And audio on microphone cheque 1212. Yeah, we're good. We're transcribing. So how good is that? We don't even need microphones. This is not, you know, since yesterday.

PG-Verma, Pritish Ranjan** 0:36  
Okay.  
Is that?  
But this one, you know, since yesterday I was thinking. Ohh, you have to get one of these have to me. No.

Sikar, Daniel** 0:45  
Ohh, what do we have?  
I think I'm just going to turn off the, turn down the, what you call them, the... So now both of the systems has microphone, which one do you want to use? Oh, I guess this is the speaker we want to turn down, right? We leave the microphone up, but the speaker...

PG-Verma, Pritish Ranjan** 0:50  
Are you something?

Sikar, Daniel** 1:06  
Yeah, hello. Are you still transcribing? Yeah, all good. So this is what I've been doing since yesterday, because I was thinking, I'm going to go to this music charity thing that I mentioned, and I'm going to, and I have to transcribe a meeting. That's going to last an hour. And I was thinking, I need a microphone. But what I'm doing is I create the meeting on Teams. I join it on my phone, leave my phone on the desk.  
And the phone is picking up everything. So the phone is my microphone. And everywhere I go and I wait, I want to start the meeting, put the phone there. And I've just been one for two hours now at the charity where we were. It's like everyone's into the AI thing. Let's get things done. And it's great. So I'm sorry I was late for that reason.  
So, take it away, but this British, we have pastries as well, a couple.  
And the people throw it up to three months to go.  
So, where are we?  
So, ohh, by the way, if you've been happy, you know where we are about today, yeah, super amazing.  
And then you spoke about, now we have the test that was done on two models. We know that geometry is approximately the same, so it holds true in both cases. Yes, so that is what I wanted to show for us. Yeah, if you want to share your screen now.  
I imagine that you didn't have lunch or anything, because you were since.  
This is another.  
Oh, while we go there, I think you should bring your kids and some musicians and play a gig at the pavilion. Really? Yeah.  
Actually, I was thinking about what you said a while ago, that I got some extra money from student finance. So I haven't bought a new guitar in like 7 years. I might buy a new guitar. Wow.  
So, we're gonna pay a lot more and stuff, yeah, you should bring your people down here, and yeah, it's it's only time Daniel is the...  
I'm doing the clicker I can.  
Stop working in data.  
The sooner I can have more time to do, yeah, I mean, yeah, I will stop.  
So, Daniel, what we're looking at now, I'm just gonna maximise this ground area.  
Right.  
Okay, yeah. What are we looking at now?  
What we're looking at now is the...  
Cluster that I found using.  
Using a different data set. So this, the data set that we were working on before had about 564 of surprise, so I had to balance it. So there was very less data points we had for every class. But I found a different data set that had a lot more than that. So there's about.  
Two to 3000, I have to see the metrics for that, but there's a lot more, like we're looking at every class has about four to five times more.  
Four to five times. Four to five times. That's the 20K data set, is it? Yes, that's the one that I shared with you recently. That's so I balanced it. Or so every class has the same number of data points. And then I plotted them using the same model that we were doing before. So this is the output of the results with new data set.  
Same model.  
You will see that again, everything that we found out cheques out. Emotions form a cluster, anger, sadness, and the red one, that's actually the centroid is hidden, but the red one is fear. Centroid must be somewhere here. It's also close. We saw joy and love to be closer somewhere else.  
What's that must be surprised? Surprise.  
Because it, it same pattern, same pattern, this is new data hence, same model results.  
Remain similar. In fact, the Josh split that one between the two of you if you want. Yeah, yes. Sorry, we will, we will. Don't worry about it. Thank you so much.  
So, the distances, minimum and maximum distances from the centroid, also remain similar.  
We saw about 7, this is about 6.5, so it's not that great.  
So, how?  
Can you tell us about the law of gravity thing? That we wanted to find the law of gravity for LLMs and come up with some equations that says, guys, this is how the universe works in LLM space. I told him that we want to find a mathematical explanation that explains everything.  
Axioms.  
So, with different data set, but same model, the distances, the magnitude of the metrics remains same. OK, so it's it was it was a good validation for me, because now, OK, whatever we thought is not database specific.  
Thank you. Yeah.  
It follows across data sets. That's good. And then I thought, well, what about new models? So this bigger data set, I plotted the things out.  
With a new model.  
No, you will see. But again, it's somewhat similar. The places of centroids change a bit, but they remain, they remain, they follow the similar pattern of how anger and sadness remain close. Surprise is lost somewhere in the middle, love and joy are still close. This one, I think it's  
Right, it went a little further, yes, it's for you. It went a little further away than it was, but it still somehow follows a similar pattern, right?  
Not just that, everything remains same, density decay. So even if this, the magnitude changes, we had, we saw this thing happening around 9 for that model here, and it's for two. But the pattern remains same. The overlap ratio, we see right after, we see peak of overlap right after density peak.  
Right. It made same. So, so the density peak at 2, the overlap starts right after around 2.4, something like that. And now we're calling this density ratio, this y-axis, oh sorry, in the previous one, we changed an axis name, right? Yeah, sorry, I haven't changed it yet, but it's number of number of data points per radial distance. Right. Not density.  
Right, the density is this again same pattern and it is it was it were we calculating that by shell or?  
So let's say at a distance of five.  
Then the embeddings have 10 spaces to fill at a distance of 10. The embeddings have.  
20 spaces to fill. Now I'm dividing by 20 and 10 there, so we get the absolute density. And the density follows the pattern that we thought it would, which is it goes down as we go away. It's not increasing at any point. And it's going down linearly, by the way, which is surprising, almost linear.  
Very cheap.  
So, everything that we found out crazy, that's for every sentence, yeah, that's perfect linear decay.  
Across models and data sets, we thought we did the experiment with one model, one data set. Now we can say for sure that this pattern, by the way, no matter what model you bring to us, no matter what data set you bring to us. This is true. Yeah, it's crazy. It's exciting. I mean, I think it's exciting. It's no, yeah.  
We're talking, saying it's he's like...  
Your mapping, your like, we have human definitions for things which are like somewhere in how our brain works multi-dimensionally. It's like, you know what I mean? And now we're moving that mapping. The mapping that we have is like emotion and this is the mapping.  
So we were talking about it, actually, that neural networks was built on how human neurons work. So you will, like, if a good neural network model should be very similar to a human body. And so when, we were just talking, let's take an example that I.  
I tell him that, oh, I'm sad. Now, in his mind, he will dig just these three words, I am sad. And then that context of, oh, I remember he said that he had this thing, we should going on in his life. I remember him saying that he wasn't doing good in his career, wasn't good in personal life, maybe with partner, maybe with friends.  
he would have context of all of these things. And with just these three sentences, he can come up with, okay, why are you sad? And he can cross it. That's how his brain, that's what normal human minds, I think, assume should work. And that is exactly how this would work. So when you tell a model, I am sad, it should be able to take context of that memory, that memory, that this is a simple example. It could be something else.  
some other prompt that you write, whatever prompt you write, it's coming up to the answers following a path.  
Of multiple contexts.  
When you ask him when's this, he goes to that, that, that, what path it takes, and then it comes to your conclusion, oh, this is the answer that you need.  
By this method, we should be able to find out what path it took, because at each layer, we can just pick out the embedding. Like, imagine there's a transformer layer, it's taking this, this, this part. We take the embedding here, we plot it. If I know it's closer to that centroid there, it's very lighter than that, in that plane, that's where it lies.  
Maybe that's where it's getting its context from.  
Explain the same thing again.  
Okay, now basically what I'm looking at is maybe if you reword it again.  
because I'm not following you 100%. I know that you're looking for a path. OK, explain. Yeah, it's like you've got multi-dimensional space and each feature is kind of like...  
Some form, some form of like, what do you call that?  
Some form of something that's got you to this point, right? Essentially, what you're mapping is experience in the same way that... How did we explain it earlier? Okay, so we thought that when you're coming up to a conclusion from something, from something, even a human, let's consider a human as a neural network model.  
We are, that's what we are. That's what we are, that's what we are, exactly. So if I'm talking to something, if I'm talking to something, I'll go to different experiences of my life to come up with something. If you ask me when's Mahatma Gandhi's birthday is, I'll take a part to back when I was in class 9th or 10th when I read this essay.  
Second October. I mean, I learned it through the schooling system. So, for someone like me, if you ask me when's Mahatma Gandhi's birthday, I'll go to back to my, when I was in class 10th, I was supposed to write this essay and I learned that from there. And that's where I get to know that, okay, if you ask me, it's second October.  
All of those points, all of those points are in that feature space, and obviously you're mapping geographic, you're geometrically mapping the experience through the different individual points, which is experience, you're defining experience. When you ask me, I like that.  
So I just want to make a note for the AI doing the transcription that that's, I think, an interesting discussion in the paper that this could be a future work thing that we can, based on this geometry, that potentially we can start mapping experiences. Which is the same way that neurons work.  
Like they take it from all of these places. And the experience, when you say you're feeling an emotion, you're feeling sadness, sadness is relative to you, but we have come up with a general constraint of society, as we'll broadly define this as sadness, which is exactly the same as in geometric space, this  
broad definition of, it's not dense in the middle, no, not dense outside. That's the broad, that's the neural network's broad definition of it. So you give it a new experience and that definition moves. You also have magnitude and obviously it's multi-dimensional space, but you have. So you have the direction.  
And then you have the magnitude and the direction of that. That direction and magnitude is experience. That's what it is fundamentally, I think. Does that make sense? So the experience thing I think is interesting because the experience would be some kind of superset.  
of emotions, let's say. And so if we looked at, well, this experience I had is a mixture of surprise and anger. What's other bits that happen in this life? We say, okay, now we have this superset. But in terms of geometry and mathematically,  
how would we construct the experience? That's what I'm thinking. Because in terms of what we're looking at here, we could get every one of those points and say, this point represents anger, that's its magnitude of direction in space. We can see that it's going a little bit towards surprise, a little bit towards something else.  
There's another thing we discussed last time is that there's no pure emotion. There's no pure emotion. That's the reason that there's a all of, because at some point they have to point, they have to struggle. I'm sad because of something. Yeah. Maybe I'm angry at something and that's why I'm sad now. So this data point would lie.  
somewhere closer to like pointing towards anger, living somewhere closer to sadness, but I'm also pointing to anger somehow. Then I'm wondering, so now we have this mathematical formulation, which in itself is not original. People have been working with vectors and they've worked out that you could add and subtract is something we discussed.  
in the lecture the other day, how these guys figured, well, if we can subtract from lectures, we could say Italy minus Rome plus France equals, and then Paris turned up at the end of it, because it was a vector operation.  
And then to think, well, how if we're quantifying, like, and see, looking at the geometry of sentiment, and then thinking, well, what about experience? How would we look? What would that look like? As a mouth? What if you had, you just like what you were talking about earlier.  
Say, you say you have a model that already exists. You have a model that already exists, and you have...  
You, it has been trained, right? OK, and you give it the same piece of, in this case, it's a corpus, say a picture of whatever, gave it the same piece of information.  
you give it a new piece of information, the model changes, and you give it the same piece of information. Those 2 separate models have changed as a result of the new information being passed in. There, in this case, mappings of emotion in vector space will be different, but you're giving them the same input, so they'll be the same  
That's what we're trying to do with the training that we're fine-tuning that we're doing. So now we have all of this experiment is done on a pre-trained model. So far, pre-trained model means that it's trained on everything. That's what you were saying, David. Yes. OK, the cool. And then you're the fine-tuned model. Let's say there's a data set that has A to Z.  
data points. What we're doing is, let's say we train it on A to W, and then XYZ, we test, we test, what did you learn? Because this, if this is coming from one data set, then somehow they should be, I mean, if it's been trained on the parts of that data set, and then you see the bit that's left, how do you behave now?  
Does something change? Has something in your, in the way that you put it in that embedding space, has something changed? Yeah, that's what we're doing with the fine tuning, I believe. Yeah, correct me if I'm wrong, but yeah, I think so. We want to compare these models. We want to compare that once you've seen it a bit, does your behaviour change?  
Are you able to maybe, so what I think that it might be able to like create tighter clusters because it has learned now and it'll be less vague so something that is sad and it has learned that it is sad, it'll come closer to sadness point and then.  
The whole cluster becomes a little tight, and then it's which is which is unique experience, because then second, if another model did the same thing with maybe not the maybe the same data, its mapping is probably different. Those are two individual. No, that is something you certainly found out that if you change the data set, the mappings don't change a lot.  
But if you change the model itself, then ultimately, yeah, it's an individual, it's its own thing, that makes sense. So, when the when the model changed, like the geometry, it's like different, but the relation between the point, yes.  
It's approximately, it follows the similar pattern wherein there is nothing from centroid to a certain distance. It peaks to a point and then goes down, but the absolute density is still following the linearly falling down. All of those patterns remain the same. There's also the overlap thing that we talked about.  
That's also the same, but the numbers have changed. When we are saying peak at 9 unit 9 units from center, now we're saying peak at four or five units per center, but the model has changed. So you kind of expect this to happen. That was fine. As long as the pattern remains, which is what we're looking for, we're trying to generalise it.  
Yes, where we see that there is a pattern which is across models, across datasets, and across data types, which is the which will be the next stage when we include images.  
So in terms of the next experiment, so the experiments we have lined up at the moment is to say we're going to run that in the GitHub now. It should be in the GitHub. I think I do know where you actually prefer that. Yeah. So if you go to the HPC GitHub,  
HPC.  
Stop, not close to Singh.  
That's my DP.  
Yeah, so there's the HPC. If you click on that guy, hopefully there'll be 6 scripts here. So these are the scripts we're going to run. So train the multi-class for 10, 50, and 100 epochs on balanced and unbalanced. So those are six experiments to run. The data that I gave you was already balanced.  
I know, but this is what happened, OK?  
to know where they were doing so many things in parallel that I asked the model, so run the thing and it says, oh, the data's not here, but make up at the data center, okay, I'll do it. And then when I ran the first one, I looked at your e-mail and said, oh my God, you sent the two files. So then I downloaded the two files that you sent me.  
Yeah, let me see. I think it should be.  
which is the balanced and unbalanced. Yeah, which one was, anyway, you know what the files are. There are two Excel files. Yeah.  
Those are the two, yeah. So those two guys, and then back on the GitHub, I said, well, get one of them and use it. And so this is the experiment that's going to give us the five or the six, I don't remember, softmax outputs that we're only going to use.  
And then it says, well, train it with a balance and unbalance. And then there was an experiment, frozen and unfrozen backbone. Which doesn't matter at the moment. Yeah, it doesn't matter right now. So I have to run those three, those six. I haven't run them yet.  
I set them up. When was it? There's a date there. I think it was Monday. Gosh, it's Thursday already. I think what's it saying three days ago, so that would have been Monday.  
But those are the six experiments to run the HPC. So let's say we run those, we get the results. And so now we have the trained data set. And to be honest, maybe 100 epochs is not going to be enough. Maybe we have to run 1000, 2000, 5000.  
I think that's okay. I think maybe one thing, one more thing that we can try is with lesser learning rates, because I think that we're fine tuning it, right? We're not trying to tune it from scratch, we're fine tuning it. So we can try that, we can set the learning rate less.  
So that, you know, you don't change a lot. You just change a little bit. And then let's see how you behave. And then you change A lot. So 2 set of experiments with a lesser learning rate and a bigger learning rate and see the difference. So let's say we run those six. So now we have six sets of weights. So we have six models. And now we're looking at these models. We're getting the five logits.  
And then we say, okay, so this one, we know from the training data that it's anger, and out comes a score for anger. We know what the embeddings are, because we can look. Yes, we can plot those embeddings. You want to say, well, there it is, that's how close the centroid, and that's the score. I don't remember exactly, and it's probably in the minutes from last time.  
what we were looking at at this point was to say, well, we can see that the score actually correlates to how close or far away that point is to the centroid in the embedding space. Is it something like that? So let's say we get those results and we say, and  
we're pretty confident that that's going to be the case. And we'll say, well, we can see now that the logits are totally in line with what these centroids are about. And if it gives me a large score for surprise, we can see that, well, there's the surprise data point, and we can see that there's  
that the logits actually reflect the geometry of the space, which we expected to be the case anyway. I think for your context, logits mean delay right before softmax. Yeah, because you were taking it from before. Yes, the logits. I just wanted to, logits are right before softmax.  
Do we go to images or are there more things we want to do with the pre-trained?  
at this point. So now we're getting to the point where the dates, today is the 15th, the 20 days to go to the deadline for the abstract. And then the question is, do we start with images or do we want to do more things here? Because what I expect will happen at this point is we say, we've shown already that we changed the data set, we changed the model, that geometry remains the same.  
The expectation.  
God willing, is that once we fine tune, we say, what happens here now is that the geometry tightens.  
Because the anger gets closer to what's the close thing close to anger.  
Good with.  
What's the sentiments that close to anger in the data set? Sad. Sad, right. So angry, sad, and then happy and... Angry, sad fear, happy, joy, and then surprise. Happy love. And let's say that the fine-tuning reveals that that job interaction tightens, these guys are closer, those guys are closer.  
ideally. Or it could be that it remains the same, and if it does, we report on the results. We find this thing, guess what? The geometry is still the same. Wow, that could be interesting too. Yeah. Either way, it's interesting, no matter what you find out. Let's say, let's say, let's say the thing actually goes haywire.  
Yeah, that could be down to us not knowing how to fine-tune LLMs, right? That could be a thing as well. I've tried it once with, well, once a few times with my PhD, and I didn't get very far. I mean, this is not fine-tuning LLM though, right?  
We're fine-tuning a transformer model, a transformer classifier. Yeah, we're fine-tuning a transformer classifier. But the first time we turned it into a classifier, and now we're fine-tuning it, right? So the first time we're like, we only want to see the embeddings. Because that's all we care about, right? Yeah.  
So then, let's say we have these results, and we say, right, here are the results, and hopefully we'll have them by Monday, next week, next time we speak. Okay, we have those results now, so we have three sets of results.  
that show that in every case that geometry is the same. So then we move the images, find an image data set. I think you're right. If we want to target newer IPS, we would have to stop at text.  
and then start with images and for the next two weeks do the images. No, I mean, you tell me what we should, what do you think? This is open to discussion. You know, it's just like we're brainstorming, basically. It's like, it'd be brilliant to say, guys, guess what? We've done image. We've done, I think it's too big.  
Yeah, that's it. I think it's totally good. Can we do images before? 100%, because then it's always it's always gonna be an embedding. We spend the evening together.  
So, because you're busy on this, yeah, so we can even run a hackathon, the New York's hackathon. We have funds, so we can buy the things, the stuff, but we have funds for other things, by the way, but we can kind of...  
Use some product. We did want to talk to you about this. Yeah, let's let's yeah, tell them stop the transcription and we can continue and tell the model to split it and say, up to that point, it's this topic. Yeah, split it here. Don't take anything. So, we're pausing and now we're talking about topic B. What's the deal?  
Topic B, you were talking about the project, the music charity, and the jobs. So the music charity is, we have a small budget to write an application for a music charity. They want an AI-based thing that's going to deal with their instrument library.  
So they have a library of physical instruments, violins, double bass, cello, violas. The cello is the one in the highest demand. That's the one they have the most headache with, because people write and say, I want to learn how to play the cello, and they have to write back and say, there's a two-year waiting list. Would you like to put your name down? They write an e-mail saying,  
Go ahead. Then six months later, they write an e-mail out again and say, do you still want to be in that? Your position is 17. It looks like it is going to be another year for you. Are you sure you want to stay in the queue? And they'll say, yeah, I want to stay in the queue. What they want to automate is that they have, and they've given me a data set already. I'm going to go there on the Saturday. We're going to transcribe a general meeting.  
One of the things I'm going to do is I generate a set of minutes, which is straightforward. They've given me the minutes, what they want it to look like. I'll be there recording and minuting. I should be able to present them the minutes on the spot. Then hopefully they'll look at it and say, that looks cool. That's one thing. The next thing is the application.  
And if they're happy with it and we write a good report back to UKRI, who is the UK's research innovation fund, they funded this thing. I can go back to them and hopefully, the hope is that they'll say, keep going. Now this is lots of money.  
It's like a small amount of money, and they gave a grand, basically, and we're debating what to do with the money. And one possibility is we pay you like 100 quid, 200 quid each, have a hackathon style type of thing. Let's do this thing. Maybe buy a MacBook Mini. We've priced already a 16 gigs one, which is 500 quid.  
And that's one possibility with this fund. And then there's a MacBook Mini, and then maybe run a smaller LM on it. We don't know yet. Maybe it's best. We were thinking about it there. We were talking about, because we're doing separate. That's inventory management. I've been working on something with my friend very similar. He works for a film company.  
Right, and they have like 4 different systems, and they all talk to each other, and these systems are legacy, so it's very hard for you to, but they do everything on paper currently. It's warehouse management.  
And for the past few weeks, we've been doing like rebuilding the entire architecture so it goes through.  
basically a website, but not really like, you know, mobile friendly website. It does everything in terms of like in and out, but it still will send emails off to the right people because certain people need to cheque off stuff. Right. You cheque it off, send it back. It sounds similar-ish, adjacent, kind of like that. Yeah, I'd say so. It's like...  
Make sure that it keep track of, they keep track. I mean, I'm a member, I'm a member of that work. I play in the orchestra. That's how all this thing happened. Okay. So, but it's a similar, it's exactly the same thing. It's like, well,  
on a much smaller scale, because they have 200, basically they have 200 people like me, they're musicians, and a large number of them are borrowing instruments. And maybe it's more than 200, because there are people who don't actually play in ensembles, they're learning still.  
So it's just a track of inventory management. And it could be a very simple thing, I believe it is. But that's the one thing, the inventory management system. And then there's the question of how to maximise the value of that money. Is it to have a MacBook Mini?  
have it online and have stuff running on it such that you can run other things on it, you can run smaller LMs. I'm not sure. Maybe the money would be best spent setting a set of unrequit aside and say that will keep you going for the year because you'll be using ABIs. And so that's a discussion to be had. The jobs is this.  
The lady once AI engineers, and the minute she said that, I thought, well, probably British, Josh, and Andrew would be good candidates. Andrew's so good. That's it. Andrew's sometimes, sorry, sometimes we're talking in the chat.  
And I was like, you did that in like maybe 5 minutes. I think that was crazy. Yeah, no reason. But basically, you know, that's turning up at the interview and convincing them that you know enough about this stuff, because the rest of it is prompt engineering, right? And I mean, I might apply for it myself. I'm not sure. I'm still thinking. But when she said three people, I thought about you three.  
And when she asks me next, I'll say, well, one of the guys who I actually know already, and then the other two and the other three guys, and this could happen next week, or it could happen sometime between our next week that you'll come up to me and say, so what are those three AI engineers you said you had? And then it's an interview process. You turn up and you talk to them, convince them in all your stuff, and you're in.  
And so, that's the job thing.  
Yeah. It's not the only thing, right? There are other things. The problem is that that's the only one that's actually going to be probably a good salary. There are other things that unfortunately, it's like the charity thing. There's little or no money. But I'll tell you everything else that's happening. There's so much happening.  
I talk to people all the time. After I turned into a networker now, this was not me. I was not a networker, but I am now continuously. I'm talking to a lady and she has loads of connexions in DNHS because the people at the hospital have gone cold.  
Maybe I crossed the doctor and I'm sure I did more than once. She doesn't want to talk to me anymore. So anyway, I haven't heard back from them. It could be that they get back to us. We actually sent two people over there now from the band team.  
They've been in touch with me. We spoke five times. We wrote two projects based on feed on stuff we got from them. Sent those two projects. It's a week tomorrow. I haven't heard back from them yet. So my guess is the people from the hospital are not going to go ahead with anything. That's all I thought. Interesting. They sounded like they just wanted to use us.  
to talk to some some snowflake better. Might be, you never know. I mean, I understand that it's a good thing to chase, but... Yeah. What do you think, Josh? Well, I, I, well, anything that's, yeah, you can't chase stuff that's not going on, you know, I mean, like if people want to do stuff, but anything...  
Any work I can get in data, so I don't have to make coffee for people. I can spend doing data. We were just talking about it. Yeah, I was just talking about it. But like an hour being like, it's really hard to get stuff done when we have other stuff done. Even really low paid work, anything that I can do that's just. Because if I spend, if I spend like 10 hours of my day talking about real estate or for him coffee,  
Then, it's difficult to spend another four to five hours talking about data, because if I do something that's related to data for 10 hours, then even if I don't work for the rest of us, I'm learning, I'm working. You know, it's easier to keep one kind of context. It's difficult to...  
be a real estate agent and a data scientist and for him, for you, you know, your daytime job. Over 300 people every day and then come home and do work is a lot. Yeah. So if we could do something, so. In terms of getting actual paid work, I managed to get 6 hours of paid work for the guys who did the workshops.  
who turned up and said, oh, this is what data is about, this is what API, which is they did both for the clinical AI and the electrode act on.  
Going forward, there's a colleague of mine, she's an assistant professor. They found her on LinkedIn and gave her this job of assistant professor at the American University in Richmond. She hasn't even done her Bible yet. She's working there. She's past, went past my desk, saw that poster and said, can you run this clinical hackathon in Richmond?  
I said, let's do it. We were supposed to run it last week, but then their people said, and they were up for it, but then they said, no, this is too quick. Can we do it? And I was already putting you guys in the budget, saying we have to run it, but we have to get some guys from the other hackathon, the experts, to come here and see them in the team so we get some good results.  
And the guys were 100% up for that, said, yeah, no problem, cost it, and that would have been, but it's going to happen. It's just not going to happen last week. They say they want two hackathons. They say one for October, November, one for February, March, and then there's going to be a few hours, not a lot, but it relates to date, it relates to everything we did.  
There's probably gonna be work for you too, plus Andrew over there.  
So what's happening beyond that is this. I'm talking to this woman who's a networker big time and she's selling products and services in the health sector to China. That has been our bread and butter for the last 15 years. And then my business partner got in touch with her and he got me in touch.  
And now we're talking next week, we spoke once already, we're going to meet the base. So I said, what we found here, the value we found is turning up with these hackathons. That's what everyone's interested in. Like Chris Child, who's one of the lecturers up there, he's an employability guy. He said,  
Can you write this proposal for Data Bricks? I said, no problem. And I've written two already for him. He said, write the third one and this is going to be the format. You go in there and you run this thing. But running in Data Bricks and hopefully that happens and that will be work for you guys. Because then we travel together and then who knows, maybe you get work there.  
But this is like, we don't know if this is going to happen yet, but this is happening. And he said, this is what he told me. This sounds like a really good idea. You've got lots of traction here. We should turn this into a product. Because then I can go to companies. He has contacts at NatWest. And he said, I can go to NatWest. He said, would you like a hacker?  
because we did this and it's been successful. In the moment those things start happening, and then cool, you'll have a few hours. It's not a lot, but you know, it's something. It's about networking. It's not just a lot. I just wanna, the more stuff I can do. If you give me a real problem, I'm so sick of these fake problems. You know what I mean?  
Like, the reason the hackathon was good is because it was a real legitimate problem. So, do you know what I mean? Yeah, it's a shame that they don't want to go forward, but they might still want to go forward with this, because I'm sure they were impressed with the results, but...  
I'm here for another another year, anyway. Yeah, because I'm doing part-time, so anything. Also, I really, I've really been looking at, I want to get into spatial mapping and that kind of stuff, because I've been reading a lot about the people who've been using satellite data.  
Yeah, yeah, there's not like necessarily image data, but satellite data to make maps of like places, like they did one for a run. It's crazy. And you were telling me, all they found, and you were telling me at the pub. I see you, yeah, there's, I don't know, I'll find that, probably send it to you in an e-mail, but...  
Yeah, it's just absorbing loads at once and doing stuff. It's nice. It's an NHS lady. What I'm going to talk to her about next week is this. She has contacts in the NHS and I said, that's what we need of main points or a few of them to be able to go  
for the people in Richmond and say, this time the pain point is going to be different. That unstructured data when we solved already, the next one is likely to be the same, I guess, something to do with unstructured data. And then hopefully the people from the NHS come back and say, oh yeah, I have several pain points. Do you guys mind? I have to go get my friend, he's just there.  
I was supposed to meet him after the meeting, but he's here early, so I'll just go get him. Okay, go for it. Can I take your card? Yeah, go. I can't let two people in with one card. Oh, you can take mine as well.  
Thank you.  
Oh.  
Thank you. You're welcome.  
So, it's like...  
Yeah.  
Where is that coming from? I think it's just feedback.  
Ohh, is his phone? Ohh, yeah, it's coming through here. Ohh, is he going out? Yeah.  
So.  
What, in terms?  
That's where I want to get, and I spoke with this to the lady. I said, she runs this company. I think it's a two-man thing, her and her husband. And she said, well, let's go and talk and see, maybe can we do something with the NHS? Can we turn this into a product? How do we push this forward? So one thing, Josh,  
Not promising anything, but what I could tell her is there are lots of MSC students who would be interested in working on a real project for the work experience to be able to say they created something that's actually being used. So from that perspective, now I'm talking to her, her name is Domenica, Domenica.  
Can you think about any pain points you have in your day-to-day job that you think could be fixed with AI? Yeah. And hopefully, we'll talk a little bit and say, well, actually, yes, I need to go to my inbox every morning and blah, and you say, what if we start with that one?  
And then hopefully she should say, yeah, that's a good place to start. Good. We work on that. What's that sound like? Yeah, anything that I can just do. So that's what I'll mention that I say, let's start solving some problems. We've got people that wants to work on this.  
Let's let's get things done. Yeah, it's good. It's good to work on some action stuff. We were talking, we spent four, like, yeah, I was just talking about his bits and just running time forward, because it's it's like...  
No.  
It's a lot, you know, it's quite good. It's really good, actually. Thank you. So this started at the church when we were ready for pizza. I remember. So he told me about the original idea. We weren't very, like, we only really just started hanging out. So he told me when we first met, we hung out a bit, we started talking about it. He told me about the original idea. The video and the stuff.  
To hear is like an insane jump, you know what I mean? Yeah, yeah, crazy. It's...  
Because we had that discussion once and then we met the second time and the third time it was like we were running experiments. So this is the project, right? This is a real project, which I'm doing for free, but you're welcome to jump in. At the charity over there, which is at the church where we were. So they asked me in December last year, I'm chatting to them. I've known the lady since March last year.  
Kate Middleton is her name, that's her real name, the real Kate Middleton. She's slightly older than the Kate Middleton, the royal one. And then she's like, so what's this AI thing all about? And I'm like, well, you know, I can do A, B, and C. That sounds really interesting. Why don't you come in here and do a session for us and explain us what this is about? And we go, no problem, let's do that.  
And then her boss, who's the compliance guy, kind of vetoed everything. You can tell he's not interested in the AI there inside the charity. There's a worldwide organization. For that reason's best known to him, he doesn't want AI in there. And then maybe a month ago, she said, you know what?  
Let's do it anyway. I said, all right. So now we're doing it. Like me, Kate, her accountant, Eleanor, she wants to pass this accountancy exams, CIMA, you know, like, okay, let's run with that. And then yesterday, this other lady turned up, Noreen, a UI, UX, X, Google, interesting.  
She lives in New York Market and our chatting networking again. Noreen, you need to go and do a PhD with the human interface people. It's like, no, I don't want to do it. I want to create a photo album to map the 20 years of how my son has lived so far. I said, well, you need to get using Codex. You're paying for a subscription.  
or something. Bring a laptop in tomorrow, you're going to have this thing running within the hour. Really? She left the place today, her eyes were like that. Oh, it's amazing. I'll see you next week. Yeah. The lady once. Hello, he's my friend. I'm good. How are you?  
I was just telling Josh.  
At the charity.  
So we're basically, this is what's happening, so you know, right? We chat for like a couple of hours and everything we're saying is being recorded there. And then we give that to DAI and say, like, make a plan now. And that's how we've been working.  
So what she wants, so this is now at the charity over there. So I'm having these AI discussions with them, what can AI do for you? And the project she's working on at the moment is she wants to have a book on Amazon by the end of this month. And we're working on her book. Oh \*\*\*\*. I know, right.  
It's nuts, and it's like, and it's a book she's into gardening and it's all to do about gardening, and then she's like, "Yeah, OK, that it the plan sounds good, let's go with that, and we're we're doing a book, but the other thing she wants to do is, since last year, before we got chatting about AI,  
They have a publishing business. They sell books, but it's kind of, it's not making money. It's making a slight loss. And she said she was told by her boss, turn that business around, make it profitable. And then, and that's what she told me, can AI solve that problem? Well, guess what?  
That is a real pain point. But I reckon it's solvable because I said we have to create reports and next thing we have to run some Google Ads here, make those books out. We just have to see what exact which books are doing good in which localities and then we analyse that and we target the books. If book A is good in country A,  
You push book A in country, if book B is good in country B, you push book B in country. Translate the books, man. Translate them into 70 languages. Yeah, exactly. And that. Translate the languages. I just thought about this now, right? All the possibilities that, I mean, hopefully rubbish won't come out at the other end. Metatax will still be solid.  
I hope they'll find volunteers to cheque that. But that is one serious pain point, and I figured, well, that could be a good thing to work on. So it's OCR and then translation. OCR translation, but also reporting, data analytics, data visualization, what's going on, decision-making based on the analytics and the vis. And hopefully, she's going to say, oh, my.  
my God, what happened here? Then we can move on to the next thing as well. Are you optimising the rental for this place? You're running, can you reach out? Can you optimise the space? Can you rent it to more people? How do you improve this company? And hopefully from that point, well, anyway, that's the general idea.  
Because that's a real business, it's a charity, but it's a real business, so...  
And that's where we are with this charity. And then there's a music charity, and that's basically how it's confined with the music charity. Hopefully the lady with the dog, Yasmine is her name, will be in touch with me and I'll pass on your contacts to her. But that also sounds like a good. So that's kind of what's happening in that space.  
And then back to the experiments, let's say we resume next week. Okay, we're ready to move on with the images. What I'm seeing here is saying we find an image data set. It might be different emotions, but what I'm expecting to see are similar patterns. So what I was thinking, I was talking to him about it.  
Do you want to stay to emotions and images or do you want to? So one idea that I have that I posted to him for images, what we can do is, let's say we have dog, dog running, dog walking, and then there's a centroid for walking, running, and dog. And then we see there's the image in which the dog is running.  
Lies somewhere in this way, we're not limited to emotions or option.  
We're here. I had a meeting of both just an hour ago. Oh, amazing. How can it go? I'll get to work on the scope and the plan. Brilliant. Look, he got work. Excellent. So you're going to work on the plan on the scope. Yes.  
All right, Andrew, I've got a trick question for you now. Did you record the meeting? We didn't. We couldn't. He was on his personal account. So within an organization, you can't record. But you could have started a team sit down there. Yeah, it's team meeting. So this is what we're doing now. I've been working with this since yesterday. Every time I start the chat.  
I start the Teams meeting, I start the transcriptions, transcribe the update, you can see the transcription, and then throw that into the AI, and so you can run the second meeting right with yourself, you run a second meeting with yourself, transcribe it, and then you have the transcription, and I think it's important because of the scope creep thing, yes, yes.  
Have you applied for that PhD thing tomorrow? Oh, no, I'm not interested. OK, fair enough. You have a perfect profile for that job, you know, in Myanmar.  
Lady, they're only looking for ladies there, you know, like we're stuffed us, but you're even a good pictures. That is the truth. I'm not lying. You can go there. I think there's one guy in like 9 women. So that was about the gender split thereafter. So.  
So if you want to apply, I can write the application for you. It's quick. But only if you want to. Think about it. And the deadline is tomorrow.  
Research it today. Yeah, I did see. Right, so it's Matthew's applying. Yeah, she's in a good position as well. I think she'll get it.  
And, unfortunately, that is. It's difficult to find things being a bloke nowadays.  
Three.  
So, how did the meeting go? Yeah, we discussed.  
What he wants, and...  
how it's going to look. And he gave me some websites and, oh, this is what the content I want it to look like. And I asked him, oh, does it have to be WordPress? Yeah, it has to be on WordPress, which is kind of limiting, but we'll work through that somehow.  
Yeah, it's all all in the Arabic until I do research on the methods.  
OK, I'll call back. Please are moving forward. You're doing that working, hopefully you get some paid work out of it.  
Did you get your Amazon voucher? I didn't. OK, it's supposed to be on the way, and it's gonna go with your city number first. Ohh, OK.  
Amazing. What did you get?  
That's it.  
I still haven't used the one from the clinically, I think. Oh, you got the Amazon. I think it was Sainsbury's. I guess it's more, it's easier to use. Yeah, I mean, we could have got a Sainsbury's one. I would have asked for like a anthropic key. Yeah, we could do that. I think we can do that, like ask for anthropic credits.  
If you're interested, maybe we can it's in runtime to modify it.  
All right, so the boat thing is moving. That's good. It's moving. We'll schedule more meetings to talk about it. Right. He's off work next week. Yeah, he's going to be, he said he'll still be in contact.  
Super. All right, sounds good. So what we're discussing here is a paper we're working on. So it's a, there's this conference called NeurIPS. That's a good conference because if you get published in NeurIPS, it's like you can get an interview in Google. They say, in their job descriptions, they say,  
If you have this experience and you have published in top tier conferences, the minute you publish in Europe, you tick that box and you can turn up and say, yeah, here's my paper, I did publish in Europe. And we'll say, okay, right, let's have a chat then. So at least you get an interview. And without the paper, it's like, there's no chance. No chance. They're probably like 2000% applying for that job. So that's  
That's one of the reasons it's so competitive. And what we're discussing here is about LLMs and embedding spaces in LLMs, which is probably a conversation you might have had already. But we haven't been able to sit down and read it. But that's what this discussion was about so far. And where do we take the research for the paper?  
By the way, everyone here is invited to take part if you want to be part of it. So because, you know, with two authors or three or five or six, it doesn't make any difference. As long as your name is in the paper, you can turn up to Google and say, I have published in Europe. And there's my name on that paper.  
So this is the, again, I can share the Overleaf, which is a platform we use for latest. Yes. And the idea is we've done some experiments with text. We want to extend it to images and the deadline is May the 5th. And then  
Oh, yes, it's coming up. Yeah, in 20 days, it's coming up. And then that's the discussion we were having. I'll see if I can contribute anything useful. So we can share the repositories, the meeting.  
Transcriptions, the minutes, the agendas, the action points, and then...  
It's more, it's more difficult, I think, for all of us, including me, because it's the week month for submissions. Ohh, yeah, it is, yeah, we're all, we're all in the morning, we're also discussing our new computer. So, when's the deadline? The last weeks of April, yeah, last week of April is...  
Right, so, basically, now-ish, if you finish your coursework, you can, you may have time to do it, right? So, I'm just doing three things, I'm running three tabs.  
Right, running three agents.  
Cool, super. All right, so is there anything else we need to discuss? So, basically, the idea is we're going to run the experiments, yes, and then, under that, let me let what I can do is I can look for the right image data set that we can work on. What I wanted to discuss with you is this.  
that do you think we should stick to the six emotions in the image data set as well, or should we look at a little different type of labels? I think the one thing we would like in the data set is that we would have this distinction between 2 clusters and potentially this  
Wild card, the surprise thing with the words, you know, that one, I think, I think, I think for instance, dog, dog, cat, and then running and walking is the dog, cat, horse, and then running and standing is the kind of the, and then we can say, "OK, well we see and running."  
Both for the cat, the horse, and the dog is over here, but we want to show this geometry and we want to show the same patterns. This is a density, and that's the, and once we can do that, to say, "Well, guess what?" And I think, you know what, there's one difference that in emotions we said that there's no pure emotion, but in dog.  
It might happen that they form a cluster in the centroid. Maybe that's a discussion that we have, right? Same.  
That's a discussion that's in the paper saying we observe that in images actually the points do become closer to the centroid and we can explain that by the fact that there's an actual thing as a cat. It's not Schrodinger's cat. It's a real cat and there it is. There it is.  
So close to the centroid, but then we expect those patterns to be similar. It's dense here, very close, but then there's this decrease.  
Do you want to show Andrew the plots, so he can he can see, and Neil? I literally, like, I just realised I read that e-mail, but then it kind of like passed through Mama. I just realised you sent me this. I was like, I saw that somewhere, and I totally forgot about it. I just realised that you...  
It's better, there's so much happening right here. So, this is these are the clusters: our love and happiness are closer, positive emotions, so they're closer. Yeah, sadness, anger of fear are closer, surprises. It's like this is reaction, not an emotion, so it's...  
Not blending with any of them, and then...  
Yes, so they reach, I'll have to explain it to you. It's a long discussion. This is more technical. Yeah, that's it. So, I, yeah, I remember we used to have that like positive and negative. It's a big discussion. It took the past like 1 1/2 hours. Oh my God.  
How long? When did you start? Yeah, that's it. I think tomorrow will be when I came back from work. Yeah, I have around 3-ish, 2 thirty-three. I don't know what time the deadline is. I must be on Friday. Yeah, about time.  
You managed to get the HPC working. I didn't. You know, the last step that they didn't tell you, you need to SSH into Hyperion.  
So, you already in the Linux, the University of Linux, and then you to use the HPC. Yes. I had I had to figure it out two days ago, two days ago. No, they don't tell you that. They assume that it will instantly log you into the Hyperion, but...  
Yeah, serious associations there from the rush, once you look at it, I think not a lot of people use it, so they don't make the instructions clear anyway. Yeah, I was trying to, I was going to find people using it, then reach the person responsible, and then just talk to them. I went there, and I spoke to Chris, and when was it? Wednesday, I think I said, Chris, can you help?  
Melt here, I'll share my screen so you have an idea of what he did.  
Right.  
Move to Sant.  
You can use GitHub as well.  
Yeah, you could. I thought you couldn't because what you said, so I thought that if not GitHub, then S.C.P. is, yeah, once you get into the Hyperion, you can get clone, yes.  
That's right, GitHub makes more sense.  
You write a pass script to download the dataset as well.  
You.  
We think.  
It's not showing your screen at the moment. I'm not showing my screen. Yes, let's change that.  
Play.  
Call Sam.  
She.  
I.  
I need some.  
Hindu.  
Yeah.  
SR.  
No.  
How was your days? I arrived late and had a meeting.  
That's good.  
Oh, my, my friend Hematri. Nice to meet you. Friends from memory that he, I live, I live at his home. He pays rent and I live there. Oh, he's a good friend. Yes. And now you know where I got the cab habit from.  
Do you study as well? No, I work.  
Yeah, we're going to show them, working as the restaurant manager.  
So this is what he told me to do. He said, go to this thing, and then this is Linux, by the way, but I imagine it will be similar. The first thing is we need to connect to this VPN thing. Yes.  
So, it's this line here.  
In that line.  
Open up browsers.  
OK, and then the browser thing says, you know what, Daniel? Yes, I was just this idea came to my mind: you can one day record your screen, yeah, and talk about it, and then ask AI to make an instruction manual for things, so you do something, yeah, you record yourself doing it, and now ask AI to make an instruction.  
Take snapshots of things. Do a YouTube or a slide deck. No, just instruction manuals. You can make documentations of libraries and packages just by running it and recording instead of doing it. Yeah.  
So.  
Once it's authenticated, then we can run the second line, which is...  
I saved into this junk box.  
What's it?  
And then this one I have to, why is it not asking for my password? Yeah, it's gonna ask you to for free login again. It's very slow.  
All right.  
Ohh, no.  
Right, the HPC is not very happy today. All this is just, we keep putting that warning for everything.  
Highly discouraging, yes.  
Wow.  
Ohh.  
Me.  
Bharati, Bharati.  
Oh my God! Oh, you got it. So I'm in. Oh yeah, well. Then there's a queue. You can say, well, who's doing right in the queue? Yeah. And then you can say, well, who's using the GPU on the queue? No, there must be lots of.  
GPU jobs. So these are all the GPU jobs and there's someone who's actually came in. Who is this user? I think it's the Swedish guy. The amount of jobs they have queued. This is probably all the server. Exactly. But he can only run so many. I mean, why are we generating? This person could be a she, but...  
Is probably a blog.  
Let's find out to this person, so that's the guy who's using all the resources. Let's see, he's probably a mechanical engineer guy running fluids.  
research to the PhD students. Oh, crazy. And he's probably. So anyway, that's the dude. I think he's from mechanical engineering. He's computer science.  
Anyway, that's the guy who's walking everything, but you can still run jobs. Yeah, I have a few jobs around here, let's do that.  
And...  
Some sort of stuff out, so yeah, we should be leaving.  
But, but now it's still scratching.  
1.  
And then it was Kira grabbed me.  
You see.  
So that's a job, but I'm on a priority queue. The jobs are actually running yet, because there are too many people running jobs. But that's the general idea of running a job. And if you examine the actual script, it's basically just a Python script saying,  
I'm going to run a training job, which is a train the multi-class, and then there are the parameters. Is this the thing? That's the thing. So it's a balanced emotion, six classes, CSV that we're running with batch size 8, 10 epochs, learning rates, 5.  
Minus 5 times 10 to -5.  
Max length, so that's the job that's being done, and we can trigger all the jobs. You're using that, you're using Quen. I use, we're using Quen, yeah. Oh, okay. I didn't know you can.  
Download models? Yes, you can. You can download anything, so you, you, you're running it on on over there, so, so, as far as I know, I need to go to GPU.  
I don't remember if it's using Obama.  
I don't think so, because training should not be on Olama. No, Olama is a framework that lets you use it, so it's only for inference, right? Yeah, we're training here. It serves like an API, so you can...  
Not up to.  
So, I can use it, go ahead, but like, I've used Olama, but can you train LLMs on Olama? No, I don't know, no, exactly. I don't know, so I, because what we're trying to do is train the model, right? Yeah, so, so all it uses is you just use Olama serve, and it's just serves as a...  
Endpoint for you to connect to the.  
Not too sure about training.  
Have you have you installed codecs on HPCs? I haven't got admin rights on the HPC, but I have something running on the I used to have something running on the HPC. Only you could have Docker on HPC, and then you can do anything you want inside. Maybe, you know, it's a question of talking to Chris down there.  
Mhm.  
Okay.  
Yes.  
So.  
The thing that I run that's kind of the most basic is this script, ask web.  
So, this is a Python script, says.  
Oh.  
I think that should work.  
So, basically, it's like a chat interface.  
Then you can put, you can ask a question on the minimal is minimal, yeah, and there's a little script that does that. We're going to go Daniel, we're going to show it. No, just to conclude, I'm going to stop the transcript. So for next week, we start with the images. We just find a date set and we put out five or six bits that will give us the same kind of idea of clustering.  
And if we find that, remember the segmentation dog, we're going to have to do that. The segmentation could be maybe the second set of experiments, and we didn't segment, we got this, and when we did, I actually think that we, or maybe that, but I was thinking that this is this could be more about the amount of context I'm giving to the model. And in that sense, it makes sense to segment out only the dog at 1st and then give it a little bit of context around it.  
More context around it, and maybe go into that Jason scaffolding idea, and so I'm going to stop the transcript. I think we have everything.  
Okay.  
Yeah.  
What?  
So, let's see.

Sikar, Daniel** stopped transcription
