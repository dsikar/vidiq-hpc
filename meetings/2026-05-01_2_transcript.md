**Paper structure-20260501_214418-Meeting Transcript**

May 1, 2026, 8:44PM

40m 48s

Sikar, Daniel** started transcription

PG-Verma, Pritish Ranjan** 0:41  
Allsmcse.  
Mm.  
Ohh nice, you have pen and duster.  
We should have done the meetings, hey. OK, so we're transcribing and good. What happens next? You're going, yes, two years, carry on. Yep, let's just finish that. That's gonna take like 10 minutes whilst you can first charge your laptop and then we talk about it. Yep, could you?  
Should I use my mic then? Yes. Is it better if I use my mic? If I use my speaker and hopefully, yes, use that. Yep, that way, I mean, if one of the microphones, I think right now your microphone's working.  
So, I'll check on two and two.  
Right, what you said came through.  
Because I think you would, I think your mic isn't working when I turn off my mic, then it's transcribing. I just turned on my mic and now it must be transcribing. Okay, so we're getting off feedback, which is good.  
But, yeah, as soon as they're on speakers, there should be this.  
OK, so it's transcribing at this end. Do you want to do you want to say something? Wait, let me check because my mic is also on. Let me.  
Yes, can you shut your mic off? Yes, now, now my mic is on. Can you see transcriptions?  
Yes, I can see you now, so if you wanna, yeah, hold on. OK, no worries.  
Yes, so right now with inside the we're inside the vid IQ HPC repository, inside experiments directory, inside brain underscore embedding underscore understanding, yeah, and that we have a folder by the name checking underscore context underscore retention underscore across underscore dimensions, yeah.  
Inside, we have done the test where we, what we try to do is from on the brain data, we train a regression model for the prediction and then using the weight of the model, we calculate the best top priority direction. We remove that direction, we train it again, and we keep on iteratively doing it.  
Tell 32 dimensions to see how 32 directions to see how these directions retain the context of emotions. This is a test that we have already done for LLM embeddings.  
And we reuse the data from that. So inside this directory, we have a plan.md file that covers the plan on how we are going to do the test. There's SRC that has the code for it and reports that has the report for it.  
So, inside AA in this.  
What we're really looking at is the graph on the x-axis is the directions removed. Can you just spell the graph title, please? Yes, the title for the graph is Recursive Context Retention Comparison of Manifold Decay, and we're looking at three.  
A line graph that has three different lines: orange, blue, and green. Blue line is for MPNet model that is pre-trained on that is pre-trained model, and we have the embeddings, so the pre-trained model embeddings, and then orange one is the other.  
accuracy retention for embeddings from Quen model. Quen model is fine-tuned on the data set. And then the green line is the accuracy retention of the brain data.  
And then what we see here is that the fine-tuned model starts off with the highest accuracy, but it steeps down fast and then reaches chance before others. What this tells us is that in a fine-tuned model,  
the context retention is more dependent on the top few directions, priority directions, and once we take that away, then we lose the context of the emotions. In A pre-trained model, the accuracy retention is more spread out than the fine-tuned model.  
So even though we keep, even though it starts off with a lower accuracy, when we keep on taking directions, it still, the accuracy doesn't fall down as much as it does for the fine-tune model. And then for the brain data, although the accuracy starts low, so which tells us that all of the...  
information that we have on the brain data is still not enough to give us an accurate information on the emotion, but it is significantly above chance. The chance for this would have would be...  
this, the dotted line at 0.2. So because the accuracy starts somewhere around between 0.5 and 0.6, which is significantly higher above the chance, we can say that there's still context retention in the 48 dimensions.  
And when we keep on taking away the directions from these embeddings, in the brain data, we see that it is more consistent and it falls down, but it keeps, it still contains some information throughout the directions.  
Ann.  
Hmm.  
After the test, that is what essentially we found out that in the pre-train and brain data, the context is retained and more distributed, but in the fine-tuned model, even though the accuracy starts at a very high rate, so it has more context initially, but...  
It's very compressed to fewer directions.  
AA.  
So that's this test. And the last list we do is for valence arousal dimensional reduction.  
So what we're looking at here is what we did is we...  
Reduce the 48 dimensions from brain and the 768 dimensions from embeddings.  
And we brought it down to two dimensions using PCA, and then these PCA dimensions are correlated with balance and arousal values that we have given to every emotion, so...  
Excuse me, we start off by giving.  
a certain valence and arousal value for every emotion. So for example, for joy slash hash, happiness slash delighted, we've given valence 0.85 and arousal 0.7. And these are the...  
Yes, so these values they have, they are given to the emotion based off of previous neural science studies.  
And then the aim here is to do that if these are the values that are given to the emotions based off on previous studies, when we do the PCA and bring down our information to two dimensions, do these two dimensions, can they be correlated with these values?  
And if they can be, then how high?  
So what we're trying to check is that when brought down to two dimensions, do those embeddings and brain data also represent emotions in this decoding, which is valence and arousal of the emotions that neural science does? So what we see here.  
is the brain Data.  
Where is this? Both.  
Joy, happiness, delighted. So for instance, we have we have 3 columns here, emotion, balance and arousal. So the first row, joy, happiness and delighted with the amount and the balance is plus dot 85 and the arousal is dot 70. Yep.  
So, that data, is that something that came from literature? Is something yes, data? No, that is something that comes from previous literature on done on neural neural science or biomedical studies. OK, so this ground truth there is.  
A citation that would back this up, yeah, OK.  
Ann.  
Then what we do is we, because balance up there is citation that would back this up, but not to the exact values.  
I mean, the citations would talk about how joy is violence, more positive on violence, and love violence is similar to joy violence. And then we can talk of that emotions like fear and anger have negative violence, which is what we've tried to implicate here. So we can't map.  
is stable to one specific. Yes, we cannot map it. It's based off on multiple studies done on neuroscience where they talk about how, so we cannot map the exact values, but let's say if surprise has lower valence than love and joy, we can back this up by saying that it has been discussed that joy and love would have a higher balance.  
Then surprise, so we can map them relationally, but we cannot map the exact values. OK, so that's a problem.  
Okay, I'll put that on paper. They this table says, and these numbers are our understanding of what this thing could be.  
Can we give it a range?  
If the range can be if if it comes from the side of the table, right, yeah.  
I think, I think we that would become a big problem because, well, carry on, sorry, I couldn't have a problem. Yes, don't worry, because it is essential for one of our major findings, how violence and arousal is attained in.  
Embedding space, alright, so and in brain space, so if those figures are there and this is kind of critical for the results, we need to find a source that says that backs up these figures. OK, yeah, got it. That's basically where we are. Yes, I understood. That is a good point out. We'll add.  
You join the meeting.  
Ohh, thank God, you, you, you need that, otherwise it would have gone come to the white. OK, yeah, absolutely.  
Yep, so this is on a three agent.  
On face multi-agent, but it, and then there's like 3 agents deep domain that was Africa venue, so you're asking, so I think Shane and it's all you were saying, shows you what is that data come from passive by some page, and then you said it becomes from various sources, but there's other scenery source.  
They would say there's a copy of this case from the source. So then the next question is, well, can we assemble that table from the data source? The next answer is yes, and then no problem, we cite the sources and we put that table together. Yes, Aimee, can you help us out with that? You're good with finding biomedical data.  
Can you help us find something that would be able, that would cite arousal and balance values, exact values or range of values for the emotions? And it might be that we don't find the exact ones, I would say an example.  
that we would expect. Yes, so we put it as an example. We put it that if we give these values.  
Example values, we we don't have to put it as exact values based off of a previous paper, then we put it as an example. Does that make sense? Well, as long as we can say we talk we got this wrong in this paper.  
We got this, this row, so there's a row here, surprise, and then there's a value, yes, arousal, then we say, well, this is the kind of range that we get, and this is found, and this has been done by these other guys. Yes, OK, sorry, I'm already in a problem with that, but these values.  
Based around, so I've got like I've got values instantly of a paper from 2023 validated paper. It does have values, but the like arousal, happiness, salience, end of fear, etcetera, but.  
They categorize different values from MSD, ICC2K, and raters, but do the values lie in a range? Can you show me the values if if we're able to find one that is similar to that makes sense?  
I hope this has happened to me before. Well, it just like I get a random circle and no matter what I do, it won't guard.  
But I'm like, what you can see, honey squeeze down my throat, it'll just last as long as it wants to last, and then eventually it'll start.  
because I have issues with my consoles, so every now and again, they just decide they're going to mess up. I have MRIs threat and everything, and they're like, oh, we should remove them, and then they don't remove them.  
We know with these, they treat arousal balance as a section. What we want is if you can find arousal and balance values for these, is that possible?  
But again, those values have to be on some sort of scale. Yes, that's what I'm saying. Scale.  
Are those ones on? Something like this, zero to one. If you can't find that, if you can find other scales, then what we do is normalize. Zero to 1 comparative to what? Do you know what I mean?  
Like you can have zero to 1 in comparison to X, Y, and Z or ZFK. Like there has got to be some sort of scale. What are they plotted off? Like, you know how we're plotting distance from centroid is going to put out a bunch of numbers. And if we're plotting emotions from distance, we're going to get another set of numbers. These numbers  
And motion balance arousal, but what is the what is the scale?  
I still don't really understand. I mean, what we need to find, Valance and...  
Let me finish the part 1st and then we go back. Does that make sense?  
I'll send you the exact values, try to see.  
So, uh...  
What we're looking at now is that we give, let's just say that we give an example that this is how valence and arousal.  
of value should range for these emotions. We give it as an example for now. As long as we can back up. Yeah, yes, okay, sure. And then, I mean, if we can't do that, then we have to remove the test and that's fine. Then we show other findings. That's it. So.  
That was the methodology we gave the other emotion these values, and then we see, let's say, if in a valence arousal map, I've given something valence one and arousal.  
One, so it lies here, and then we try to see that is this kind of mapping similar when we do after PCA we bring down the dimensions to two for brain and embedding data.  
And then we try to correlate that, if not for both axis, which axis is aligned with, so we we we check the X&Y axis here with both balance and arousal, and we try to see can they be correlated, and then what we found out after reducing.  
Yes.  
Where is the report on this?  
Koirala.  
No.  
Shwe.  
I had an HTML file for this as well, but it's lost. Okay, so can you just describe the HTML file we're looking for?  
AA.  
No, it's not her.  
Okay, I'll look for it because I still have the findings.  
It's just it's easier to look at it when it's an HTML file.  
So, what we found out is that, uh...  
that when we, by the way, we did this test for Quen 768 dimensions, which is a fine-tuned model, MP net balance, which is a pre-trained model, and human brain data. And then we found out that PC1, which is one of those dimensions, showed 0.98.  
correlation with balance and PC2 in that case showed 0.82 correlation with arousal.  
I did it with both. So I mean, the only reason if it showed, I'm showing the higher value. So PC one was also matched with arousal and then just the value came out to be 0.1 or 0.0. So it was a very small, small correlation. That is why I'm trying to show the one that it had higher correlation with.  
So you had balance and arousal and arousal consistently with the higher value between the two. Is that what you're saying? No, what I'm saying is that.  
Let's say I have balance and arousal here VA and then they have LLM, PC1, PC2, PC1, PC2, right? Also have brain.  
PC1, PC2. I calculated this with this and this with this, the correlation between both of them. And then I just found out that PC, one of them was higher with its PC2, was they had 0.85 or 0.9 or something, while PC2 with balance had 0.1.  
Then, I'm just showing, and right now I'm showing these this value.  
So you found there was a stronger correlation between what you called PC1 and what happened is that in LLM data, one of the dimensions showed higher. So if let's say PC2 showed higher with a greater numbers of 0.85 or something, then PC1 showed a higher value for balance.  
and a lower value for arousal. So you look at a series of values for one, a series of values for the other. So what I had at the end of it was for table PC1, PC2, balance and arousal. And so what happened was PC1 would have a 0.90.  
0.1 0.88 0.21 0.2 0.88. So what I'm trying to say is that then this dimension might is have correlation with this and this dimension has correlation with this.  
Mm.  
Nothing against.  
So are you looking at 2 distinct? So let's say we're looking at two. I give you 2 tables and I say, here's a table that has a column A&B, and here's a table that has a column C&D. You have to map these two tables to each other and tell me which column  
to the column in the other table and you're saying, well, I'm going to inspect the values one after the other and saying in this linear sequence how they correlated they are. And that I did that. What if we randomized the sequence?  
I'm not checking row by row, this is what I'm checking, that you see this pattern. So one of them is ground truth, that means ideal, and another one is what, I mean, the ground truth is right now we still look, we are not sure about the ground truth because we don't have back to back, but what I'm trying to say.  
is that you see the pattern, that surprise up there, surprise here for both of them, surprise, surprise. Then there's joy, there's joy here, there's love right below joy. Similarly, there's love right below joy, the sadness, sadness here, anger and fear are there, and the similar kind of representation. So what I'm trying to map.  
is the fact that...  
The geometrically there, there's a pattern. So what we're saying is, if we get balance and arousal for brain data and plot it there, and then we say, OK, there's the balance and arousal for anger.  
and it's up here. And now we have anger for our LLM data. So now we want to plot it in here as well. So we've done the dimensionality reduction. We ended up with two coordinates. And then you say, well, it's X&Y. But now we can flip it, right, and say, well, it's PCA, PC, which one's going to be X, which one's going to be Y? And then you decided.  
If we flip it this way, it's going to be closer to that one over there. Yep. And if we have applied the same flip to everyone, and then they're all going to be close. And the same flip is going to apply to everyone. Yep. OK.  
So.  
So all of those, if let's say, if I'm checking PC1 for anger, then I'm checking PC1, let's say I'm checking PC1, I'm saying PC1 is valence, then I'm not just saying that that's the case for anger. I'm checking for all of the emotions that PC1 of all five emotions is checked.  
with balance of all 5 emotions. So now in red, we have aligned embeddings, in blue we have ground truth.  
Yeah.  
So, what does that mean?  
Yes, that means that the red crosses are PC1 and PC2 values that come from embeddings. When we do PC reduction, sorry, dimensional reductions on the embeddings.  
And then the blue dots, which is written here as round truth, are the values that from the table, from the table. OK, so the values that we need to back up, yes. OK.  
And then what happens is that we say they are highly correlated because there's a geometrical presence in the space is very similar.  
Ann.  
However, for brain data, what we found out that only arousal...  
Arousal is the one that we can correlate it with, so...  
I mean, the balance may not match, but the arousal values do.  
What's the metric that you're measuring? The isn't that correct reason that they prioritize different axes?  
Yes, so say that again. They prioritize different axes. So, for example, the brain prioritizes arousal, yeah, and LLMs prioritize violence. So, essentially, they're working up the same metrics, but the...  
Like, if you put valence is X, arousal is Y, yeah, that's how LLM works, whereas brain works, arousal is X, valence is Y. No, no, no, no, no, no, no, no, what we're trying to say here is that when brain, when...  
Brain is storing the information in those 48 dimensions. It's storing the context about the arousal of the emotion. Whilst when LLM is storing context in all of the 768 dimensions, it has the context of the arousal and the valence, which can be seen when we do bring it down to PCA that they correlate.  
You can't say that because the brain also has arousal and a balance. Yes, it's a smaller proportion. Yes, we do not say at any point that the brain does not have balance there because the brain does work with balance and arousal. Again, they're just prioritizing different axes.  
I mean, it also, we cannot even say that a human does it, that human brain does not store violence, by the way, because maybe it stores violence in some other manner. Maybe, you know, it's based on the data. Yes, it's based on the data that we have, that these activation values may not contain.  
Yeah, the violence information, just prioritizing them geometrically, differently is the correct wording, I suppose.  
See, they both have both. It's just that they're prioritizing them dramatically differently. Yeah, OK. Brian is prioritizing arousal and the land prioritizing things.  
I mean, this also backs up our talk, how we found out that anger, sorry, negative emotions like sad and anger are far away from happy and joy, because their valence are very different. Valence is the measure of how positive or negative an emotion is, and arousal is a measure of how, yes, intense or how...  
how stimulated you can be by that emotion. So, I mean, in that sense, anger and surprise are because both the more intense emotions, they might lie closer in that space, whereas that is not the exact thing that we found out for LLM data.  
So what we're trying to say that in the geometrical representation of emotions for brain and LLM, brain would have clusters of similar arousal values closer. However,  
In LLM space, the...  
You.  
Valence is the determining factor. Yep, valence is the determining factor, major determining factor of how close or far away an emotion is.  
And then the statistical validation. Yep, I agree.  
Currently transcribing, yes.  
Okay, so.  
Yes.  
Measures.  
I'm not going to share the screen. I'm going to input this all into a singular document. If I can bring the document up, I'll try and do that as I'm speaking. Okay. So the table looking at a motion, valence and arousal is a table that is based of.  
an original journal which is significantly cited from the 1980s by a J.A. Russell. This is a so complex model of effect. It has been significantly cited since as a fundamental  
Yeah, fundamental work looking at violence and arousal with emotion in the brain. More recent data sets that are cited using the same scales are DEAP, a database for emotional analysis.  
using psychological signals, and this is by authors Sander Kolstra and Christian Mule. Again, this has 5245 citations, so it is a well documented piece of psychological.  
I saw history at this point, yeah, Research history.  
Super.  
So we have two citations that back up the figures in that table. Two parts.  
Sorry, so one is the J. Russell, and the other is the Santa Claus and Christian Muir.  
Yes. And these will be in the folder previously mentioned, which is LLM slash brain. And the exact document is LLM underscore brain dot RTF. This will contain those two references plus  
around 10 to 15 others that are relevant for this piece of work.  
Super, so that's all transcribed through clicks, get your text done, but let me know we have a lot of sleep in the break just to get it in there. Take care.  
Huq.  
OK, so Aimee found all the.  
On the two citation that back up the figures on that table.  
So, we're good, very good. Yes, so. Oh, we can't say that they're fundamentally accurate. We just say that they're heavily documented because they've never been proved as fundamentally accurate. They are just significantly documented because the theory. I think it's also because these are values that are given by humans to the how emotion, you know, how human brains reacts to emotions and neural science.  
they gave these values. Okay, we see that there's certain things that are higher for these emotions and there's certain things that are higher for these emotions. And so these values are not actually physical representation of something in the brain. These are values that are just used to decode an emotion when we want to look at it for mathematical operations or any other.  
you know, task to be done on the motions. So it's like a labeling. Yes, it's like a labeling. It's like a labeling. It's one of those theories that was free in a long time ago. It's never been able to be proved incorrect because the theory withholds. And then also it's one of those things that no matter how much scientific work has been done.  
We cannot physically also prove it adequately correct, because we're working with brains of emotions, and people have tried to understand it for years, but they can't, but, but, but, but actually I think it would be really nice to put it in the paper now, because if let's say 10 people have seen the, ohh, emotion can be decoded in this way, and then we say that.  
We put another paper which also say that, even if you look at it as embeddings, the the the the yes, the premise still retains the geometry holds. It's like looking at the universal part of intelligence, but no matter what, you can't explain it is that, and that is the patterns that we are seeing.  
And then we just say that this happened. Right, that we're going to come up with some universal law of geometry, and if it applies to both geometry of LLMs and brains, wow, amazing. And that's why we need to keep the storyline. That's why we need the reproducibility we were talking about earlier, because this is a study that can be moved across animals and things like that. And if the protein consistently reproduces, then it.  
points for a universal part of intelligence, so again, nobody can necessarily claim it as such. It's just saying the system, yeah. All right, so that's looking good then. Yep. So do we?  
discuss the structure of the paper. Yes. Is there any other bits you want to show? No, there's no bits. I'm sorry about that call to my friend. She's has some medical tests going on. She just found out that she's getting blurry with one of her eyes. Oh, sure. Like, yeah, a few hours, few hours ago. And then, I mean, she was about to go to sleep, like it was 8 P.m. for in my home country.  
And then it's 9 now and for since like one hour, one of her eyes, it's getting, she has to keep her eyes shut. I feel she goes under too much light and one of her eyes vision is getting blurry. Oh, \*\*\*\*. And if it has to be some strong infections, so she just has to get home and get, not.  
People get home, like, get up in the morning, and first things first, we go soon to get that checked. That's why I had to take the call. Sorry, but I was cycling home and I lost my way, and I didn't know where I was, and then I went back to the rowing club where I left, and then I retraced the route, but I don't remember. I got to the back. Mind you, I shouldn't be transcribing this, 'cause...  
I want to pull the transcription anyway, so what happened basically was that I got back here and then I...

Sikar, Daniel** stopped transcription
