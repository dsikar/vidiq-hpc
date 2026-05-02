**Image Embeddings-20260428_193652-Meeting Transcript**

April 28, 2026, 6:36PM

1h 44m 42s

Sikar, Daniel** started transcription

PG-Verma, Pritish Ranjan** 0:29  
Yeah, are we transcribed from that? Yes, so the room module is coming in.  
So now, in theory, if we mute ourselves, it should be transcribing.  
Give us one more line.  
Yes, we're good. So today we're transcribing through a single microphone, which is the room audio. We have in the room Freya Myo, Andrew, Amy, who has a background in biomedical neuroscience. Did I get that right? You told me I forgot.  
Biomedical Science, yeah. Biomedical Science. She brought a neuroscience, she's bringing a neural science angle into the work, into the paper. And it so happens that the image, the sentiment image data set, 120K, was very noisy. We're not getting good cluster separation. The British and Amy had a chat.  
about her work and it looks like biomedical neuroscience and the related data sets is the way to go. And also President Arj, me, Daniel and Pritish. There's a single microphone, so AI, you're going to have to separate all the voices by style, etc.  
So what we discussed so far is Amy introduced herself and told about the research and what and the interesting findings she had. So Amy, could you please repeat it? So we have a record of the transcription of what you said so far. Is that all right? Yes, that's fine.  
Oh, it's so much pressure when you've got to do it this way though, isn't it? I'll remind you. So you were saying that you got chatting and then you had some data sets and you saw that in the data sets you had that were based on MRIs, you could see that certain areas in the brain were activated depending on the sentiments that were  
and of that the people who were being scanned were experiencing. Is that was that something in those lines? Yes, that's correct. There was around 48 different vectors which are mapped into the 48 dimensions. So I guess. Question of 48 dimensional vectors. Yes. Yeah.  
So essentially, these represent different areas of the brain, like the amygdala, hippocampus, insula, prefrontal cortex. And what it was looking at is when people are experiencing those set of emotions, what areas of those brains are lighting up?  
and having the most influence. And those can then be mapped to numbers. Once they're mapped to numbers, you can bring them down through the means or whatever to the centroid. And then we've then used those centroids to map.  
what is happening between the LLMs and the brain and whether there's any similar patterns between the two. And to clarify, so what the process you described of getting means and centroids is something that happened after you started discussing this with Pritish, when you expert, where this is published.  
Work that exists prior to this conversation.  
No, this is a data set that I took and ran through after the conversation with Pritish because I wanted to see if what I thought his data set looked like, if it actually matched with. Amazing, and this is the work you believe could be original. So if it's presented, it says, well, this is something that's.  
Like people are thinking about studying, but there's nothing kind of obvious that it really. OK, super nice. Particularly with the ambiguity gradient, that's something that's not previously been looked at before, and that's our strongest figure. And what you, the term you mentioned, ambiguity gradients, can you explain like in?  
What's the term? I don't understand it myself. I never came across it. It could be something that I know, but through other intuitions or other terms. So it essentially means that in both the brain and within the language learning models, if  
If we remain and the figures are close to the centroid, then we've got a strong, emotional, consistent network engagement that's going on. So a clear emotional state. And what's then happening is as you move further away from that centroid, you're getting mixed network signals. You're losing that ambiguous perception.  
So it can no longer clearly categorize the emotions when you move further away. So if you're moving further away, you said you're losing the ambiguous perception. That's what you. So as by moving away, you're losing the ambiguous perception by approximating, you're gaining the ambiguous or.  
Is that how it's working in terms of correlating? I'll just jump in what we so the first of all I'll just do refurbishment 48 dimensions between them as embeddings and we scale them down. That's also important because these numbers in the 48 dimension rod.  
Or, when we take it raw from brain brain data, they're very, you know, it's not properly scaled, so we scale them down to, I guess, zero to 1, if I'm not wrong, we scale them down, normalized, yes, we normalize it, and then after we've normalized it, we log these points from 40 on the forty-eight dimensional Cartesian plane.  
And then for visualization, we bring it down to two using PCA, but we brought them in the 48 dimensional space. And then the first thing that we looked at was the density, I call it density decay again and again, but essentially it was number of data points per radial distance band. Right.  
And then there was a lot of similarity between how it grows to a point and then it goes down. Although because of one thing to be mentioned is that the brain data that we are working on right now, or we worked on before, had very few subjects. So it has 40 people that have felt five different emotions, which are not exactly the same. So there's a difference because  
It is depressed, delight, excited, and I can have a look, but I don't know if the software has. So, yes, these are the emotions, 5 different emotions that we're working on, and every person so depressed, delight, excited, three, those are three out of five, afraid, calm.  
delighted, depressed, and excited. So I imagine then we'd observe that afraid and depressed might kind of be, no, no, no, we didn't. We didn't. Oh, that series of the brain, right? Yes, so it's a game. We're getting there. So what happened? First of all, we did the density.  
The decay thing, and because the data set was much smaller, so it was very irregular, and we can show the graph to you in a bit, I think. I think you pull that one, I think, yeah, and show it, so that was the first thing that we tested.  
I actually have a paragraph here that I typed up that runs through the ambiguity gradient quite well. Can you read it? Yeah. So the ambiguity gradient measures whether emotional uncertainty increases as brain activity pattern moves away from the prototypical pattern or its labeled emotion.  
In our case, each emotion centroid represents an average distributed ROI, which is region of interest, activation pattern. Samples close to that centroid are more typical and easier to classify, while samples further away are more mixed and ambiguous.  
This suggests that emotional ambiguity is not just noise, it's structured geometrically in the brain's representational space. This is an important cross-system result, is that LLMs show similar ambiguity distance relationship. So even though brain and LLM may differ in global category structure and redundancy,  
They appear to share a local geometric principle, which is uncertainty increases with distance from emotional prototypes.  
Which is something we found out in our test as well. We found out that the centroid of an emotion was not the most anger, but the purest anger, which is similar here, that the ambiguity in the emotion increases when you move further away from the centroid.  
So that is a similar pattern between that behavior in LLM and then so this is the graph for the density. You might remember this one, this is what we usually saw about, this is the pattern, the dotted pattern we saw in the LLM for every emotion. So this is when we look at one emotion.  
and we look at the number of points, how they increase or decrease when we create bands at radial distance from the centroid.  
And then there was about the bands, so the bands are those guys there, 20, yeah, 25, 50, 75, OK, radial distance bands, and the number of data points between them, more like the shells things, yes. So, if I was to read, if I was to take like one data point out of there, so human brain is...  
the PDF.  
PDF. What's PDF again? I think that's a nomenclature that AI use that does not have a space. Is there not something between like CDF and PDF? Like CDF? Isn't it a statistical term? Anyway.  
So, to be, to be, um, yes, so...  
If we get one point, say one, and we go one.  
parallel to the y-axis and we hit almost the kind of the peak of both of them.  
If we go up this guy.  
You get that point in that point, the last defensity here is the highest for the transformer.  
Yeah, because so most of the examples would fall more or less at this distance from the centroid. Yeah, for the transformer one. Yeah, I think that's also because the data set for transformers, there was a big data set. And this has a very small number of data points. So I think when we are able to add more data points, then this would  
Have maybe not go that high, but have a similar, but that's like one of the smoother something for the discussion, right? The same.  
Because there's less data points would explain why it's not as smooth. So, another question is this: So, is this the scale normalized to say we want the same radius for both this, because they...  
So, we the clusters are formed in two different metric dimensions, so we had to scale them down together to plot them one over the other. That is right, and then we tested. Can you do then you talk about how we tested the?  
I'm more on the statistics, not the figures, so I'm not sure what one you're asking for. Oh, you got it.  
So, we tested, we tested how the two density decay graphs are compared.  
So, one ambiguity gradient similarity, correlation, P-value insights, high correlation indicates both systems represent uncertainty.  
So that P-value is a near 0 chance. So like a near 0 chance that it is chance, if that makes sense. So it means that they very strongly and we did, I don't think you've got the figures there, but we did a bunch of permutation tests, shuffle tests, so, and yeah.  
Means we we verify that that figure is absolute, and so just for the transcription, all the stats you just described are in which file here or in which folder. This is where the AI right on this write-up happens to be able to say we're writing up based on what you just said there.  
So, is this...  
comparison result.txt. So once you just described, you just said one paragraph and you said you tried several tests and it shows that the uncertainty is very low. You're pretty sure that there is a wrong. You should have that on yours. Remember we ran a bunch of the statistical tests, the permutation, the bootstrap.  
the low sorry, you should have all of those on there somewhere. Yeah, so.  
So, we have a hope when the time comes to write this up, and it might not even fit in the main article now that I'm saying this could be all appendix, or some of it would have to be put on appendix, which is great, because like article with a large appendix, whoever's reviewing, these guys did their homework, so it's a whole thing.  
So where are these physical analysis that Amy just described? Is it? Are these? Is this actual written as text or is this something else? That's what you did. It's not a problem. It's not written as text. The main thing is connected with something here. So this is something that was ran because I don't have the.  
LLM data set, right? I've only got the brain data set, so this was ran on Pritish's computer, so he has all of those figures on here somewhere. He's just...  
This push.  
No.  
But the main thing is, what you just said, where is it? Yeah, but then when we do the transcription, ohh, yes, then we know and say, OK, I know where those files are and we can connect what Amy just said to an experiment. It's inside human brain emotion exports. OK, so on global behavior comparison, we have comparison results.txt. OK, so just going over.  
The structure, we have video understanding, and then from video understanding, we have underneath it human brain emotion exports, and then under human brain emotion exports, we have cross-system results, and then on the cross, is that where we are? No, this is where we are.  
Global behavior, so under human brain and motion exports, we look at global behavior comparison, and then we look at those text files. Is that is that it, Amy? Does that make sense? Yep.  
Yeah, I can see the Pearson correlation. I can see the P-values there. So yeah, that will be our...  
OK, so the statistical robustness, essentially making sure that our results are valid, because we need all of that for anything to be publishable, basically, and you mentioned another as well. I didn't get it by name. I don't, I'm not that thought on statistics. It might have been the low so or the ohh yeah, low so yeah, I believe that was.  
in there as well. We were running a ton of stuff yesterday and you've just reminded me I need to export this as a CSV to send over to you, Pritish, so that you've got. So the new Amy is going to export results. Is that right, Amy? It shows up as an action point in the agenda.  
Yeah, so I've got low so on my end here, but not for these cross-validation tests. What I have is the low so, the permutation, the bootstrap and everything for purely the brain data, so that we can say, look, the brain data itself shows that these signals are completely valid.  
I'm not there significantly above chance, blah, blah, blah, and we're going to need that before we do the comparisons, but Pritish has said everything in terms of the comparisons. So I'm just going to export this as a CSV and send it over to him. So you've got all of the brain statistics as well, just to summarize and so we can show that this is solid. Amazing.  
Well, I'm sorry if I sound a bit bossy, you know, it's just that sometimes I'm like that. No, you're good. I was doing the same with British yesterday, going British. I know it looks good, but statistics, we need the robustness, we need the generalization and that.  
Me and Josh have to do for the next few days, whatever we worked on before, but Josh, you have the contents or whatever, you have to do that on a bunch of data sets, and yeah, no, don't worry, I'll explain the reports. No, no, I said I've got some anyway, we'll go.  
All right, that's looking amazing. Do you know, even that 120K could be appendix stuff. Let's say we tried it on 120K, it's very noisy. Here's some data. I think that's a good thing to show as well. Oh, that's actually another note. You keep mentioning that the...  
brain data set that we used is small, but when I've looked at a bunch of published studies, including ones published by where we're submitting this, a lot of the...  
Well, pretty much everything that's been published uses maybe eight people, like 10 people, because brain data is so messy and because you're taking so many snapshots of like every millisecond of the brain, essentially, we've actually got a really solid subject base. So this is a good amount. We have 40 people and five emotions per person. So I'm just, I did not know that if that is true, then that's good.  
Pretty good day. Well, we had 40 subjects, 40 subjects, and five emotions per subject. Yes, correct. That is actually a big data set in this area. Is the reason why 100 for cross generalization anyway, so the reason why I said the disparity is because in LLM we have like 20,000, yeah, yeah, but I said that's a...  
That's an important thing for the paper. So what you just said, that studies, that for studies in this domain, the data set is actually of a good size, because the referees who don't know your domain are going to say, oh, but come on, guys, look at the size of your data set. And you say, well, actually.  
It's a pretty good size and it's a good kind of counter argument of front load and say, don't, there's no point in asking about the size of the data set because that's a standard size in this domain. It's above standard, in fact. And then the thing then is, is that the kind of thing, like if we pinged you, because we have, I think, seven days to go.  
The deadline, but there is not much time left. Well, but if any results are all there, it's good. Yes, the results are out there, and I think Amy has done a really good job with the quantification and the statistical, you know, weak strength of it, but me and Josh would have to work on the bits that we did earlier.  
We did, we did look at multiple data sets, but we have to quantify them now with silhouettes for the clusters and proper numerics. You know, we have to say that we tried with this and this data set, this and this model, so statistics, or yes, the statistics of it. See, that's the good influencer.  
of that you bring into, we need statistics significance in built in. Yeah, 100%, because otherwise you could find something amazing, but if they can't go, okay, well, this is robust, this generalizes, we can see that this is statistically significant because they have done all these other tests which shuffle the values, do blah, blah, blah, they'll throw it out.  
Like, it's super, super, super important, which, again, generalization, I've said obviously, I think I've been lacking there, because we look at the, I've been lucky too, right? Look at the visualizations, we start assuming next set of experiments, but we didn't quantify the previous visualizations with enough.  
statistics and proper numerics too. But anyways, so what I was talking about, I'll discuss what I've done before and then we can get back to more findings. Or do you want to finish that first? No, no, no, you can carry on where you are. I'm just getting the CSV to send over to you quickly. So Pritish, I think it would be a good thing.  
I'll ask questions as you go along, yeah. So for instance, you're going, you're about to explain a bunch of plots here. So what I'll be asking is where in the directory structure that you're going to push the GitHub these plots on? Because when the time comes for the, for our model, or for our AI agents to do the write-up, it will know where those images are.  
I'll have to push it, but no, but yes, right now what I'm what I've done is, so I'll explain with the directory structure of it, yeah, inside of VidIQ HPC repository, inside experiments directory, we had text directory embedding field and text model.  
Where tech inside text, we were working with the text data inside text model, we were working with the custom model for the, you know, custom LLM model for the classification inside embedding fields, we have we have text and image, and I think you've also pushed.  
image inside the experiments, which I haven't pulled so far, so it's not there. Okay. But yes, inside embedding fields, inside text, we have binary and multi-class directory in which we've worked with binary data set. By binary text data set, I mean the word and labeled positive or negative.  
And inside multi-class is where we have sentence and labeled the emotions, one of the six emotions that we were working on. And this is where inside that, then these directories discuss, by these I mean balanced data set, and the directories inside the multi-class directory discuss the different models and data sets that we worked on.  
So what I did was inside the experiments directory, I added another understanding text embeddings directory, where I've planned out a set of.  
Experiments that...  
Essentially, we're just so what I wanted to do at this point was, because, because at first I did the image embeddings part, but the clusters weren't good. I did not see proper clusters formation in emotion data set. I did not see any clusters in the animal data set.  
I did see a cluster, but it was very difficult to make out something. I'll show it later, but for now, let's just say, so I decided what to do now. So I thought.  
And so, in the text embeddings, although we looked at the cluster and the and we looked at the visualizations part of it, so where it is being plotted in the 768 dimensions and where it exists on that Cartesian plane, and we when we came and we came to some conclusions on the basis of that, but what we missed is to look at the embeddings again.  
back again from the plots. So the embeddings exist in 768 length, right? But...  
But I thought that not all of the 768 numbers would be holding context of just emotions, because these embeddings, when they created out of the text, at least the pre-trained models, no, the pre-trained models with the generalized models, their job is much faster than just emotions.  
They're holding context of everything in the text, not just emotions. So the emotions, I assume that the emotions might exist in some of those dimensions instead of all 768 dimensions. So the next set of experiments that I did was to test that hypothesis of one.  
And yeah, I know things.  
So first thing, I plan it on structurally. So first thing I did was plot the clusters again and look at the silhouette scores. So this table we're looking at, unified phase one global geometry redo. So you have intrinsic emotion geometry across 5 LLM configurations in human brain.  
MRI data with L2 and Z score normalization.  
So, battery level, do you want to do you want me to?  
I think you're a battery low there. Yes, I do remember.  
Is there a block point? Don't worry, there is a block point nearby, and I do work before. Gotcha, and is it gonna? Is it gonna? Yeah, OK. Super.  
The.  
So, so L2 is that score normalization? Yes, I ended up doing normalization on the embeddings, and I tell you why, because when we're looking at these normalized embeddings, normalized embeddings, at first I thought that the embeddings, if we keep the magnitude of those vectors.  
We're also storing the intensity of the emotions, but then these embeddings, they are not just about emotions. So we cannot assume that all that the magnitude holds is the intensity of the emotion. Yes. That's really good. I like it. Good thing that we're transcribing that, because I imagine that goes as a justification.  
for a referee, say, why did you normalize that? And we front load that and we normalized it because it is believed that that's the embedding doesn't hold that emotion alone. It holds all sorts of things. Okay, so it's like a justification that we carry on.  
If you're not getting all of this, if you, I am, well, if you, if I have enough, if you have it, I'll just do well, because these are the, you know, I will, yeah, yeah, hey, you guys are pressed for time, sorry, you need to go, go for it, we can continue like online, or I'm not pressed for time, sorry, I'm also just very...  
I have time, I will. So, an hour late, and no worries, you brought cookies. So, configuration, those are the one, two, three, 4, 5, 6 configurations, and the last one, the last row is a brain MRI.  
Yes, that's the that's the one that Amy helped us find. What, what, what is Sin you at?  
Silhouette score is how tightly and how properly shaped the clusters are, right? So, is that a thing? Yes, it's a thing. It's a main metric, and it's a metric for for clustering. OK, and when I see that it has a negative.  
Ali.  
I'll firstly explain all the...  
All the data points that we're looking at, right? So, for the transcription, this report that I'm looking at exists in the understanding text embeddings that I was talking about inside. Say that, so the reporting we were looking at, can you go back to the reporting page, please?  
So the reporting we're looking at is a page that the header is unified phase one, global geometry and in parenthesis redo. And on the file system, can we go back the file system? It's under experiments. So this is phase like UHPC experiments.  
understanding as embeddings reports, phase one summary of all the HTML. So that's the phase we're looking at, cool.  
Yeah.  
So, this table.  
The data, the data points that we're looking at, these are embeddings. This is the, so we look at, we trained, we had two models initially that were used to get the 768 size embeddings from the text, yeah, PG and MP net, yeah, and the DSA are.  
The decide I or whatever you want to be DSKIRAI data set, yeah, 6 emotions, fewer data points, and the balance data set has the same 6 emotions, but a lot more data points, and also the first one, PG balance.  
Is the embedding 768 size embeddings using the PG model on the balanced data set?  
The second one, BGE, SRI, is PG model, SRI data set, 768 size embeddings, both of them pre-trained, so they have not looked at the data set before, and we net, because I found out that this particular data set, SRI data set, had very few data points and was very noisy.  
I decided to use the verify the MPNet model on just the balanced data set runs, and then when native and when 768.  
Uh, native on the embeddings after model was fine-tuned on the balanced data set on the six emotions, and so essentially the native embeddings that you had provided after the fine-tuning were of 2048 size and not 768 on now.  
Yes, so it must have been an experiment. No, you don't have to repeat it. I did PCA on it. OK, so I did a PCA on the 2048 sized embeddings, bring it down to 768 right wage.  
2048 you said. Right, okay.  
But anyways, I use both of them. I just wanted to verify, are we doing something when we go from 204 into 768? Because the 768 does not come from the model itself.  
I wanted to verify that we're not losing anything when we do BCA.  
And the brain FMRI data set is the one that Amy was talking about. This has 40 subjects and all of the 40 subjects, 5 emotions and how their brain, which portions of their brains activate.  
When, when they perceive or react to a particular emotion.  
Yeah, we're looking at 48 different points inside the brain and the activations in them, and we bring the when we consider that list, the scaling has a 48 size embeddings to be able to compare for the passing.  
Um...  
I plot these embeddings in their dimensional space and I do BCA to visualize them. So in the three, Amy, can I ask you a question about these 48 dimensions? So do they map to an actual area of the brain or is this?  
Yes, they do. Yeah, so not on a...  
Neuron level, so not on like a, because that's like millions and millions of networks, but on a...  
pro level, they are mapping to portions of the brain. So, like, you know how I said earlier, you can look at the prefrontal cortex, the insula, the hippocampus. It's mapping to those different areas of the brain that are being activated when people are perceiving. So, like, what I'm thinking of is the brain as something that exists in 3D space.  
So now I'm trying to translate it and I haven't got the tools to do this, to translate this into 48 dimension, the 40 dimensions. So are we talking, are we saying that while one area is being activated, there is also some level of activation in another area, but primarily one area is being activated more than  
Another of the. I think we should. Yes. I believe this has all of those. Right. This has the 40 columns here that say these were the activations in every. Yeah. So with the subject, you've got all of the different areas of the brain here.  
and you've got all of the different numeric activation levels essentially. So these are all happening. Concurrently. It's concurrently. So one subject when he's upgrade, these are the areas in his brain and these are the activations and those. Information about all the areas in the brain for each emotion.  
Essentially, so that would match, so that would be the four T. I mentioned, what do you need to 40 damage super, yeah?  
And then I plotted them out along with this silhouette scores, because, as I have one more question, that CSP is the data set, essentially, yes.  
Oh yeah, so cool. Sorry, say that. The one that I exported and sent to you, wasn't it? Yeah. Not the original data set, but that's the CSV that's been original data. So the numeric things for him to be able to. Sorry, I saw just to make a record of this in the transcription, there exists an original data set.  
The data sets that Amy worked on is was derived from the original data set. Yes, I exported a bunch of CDs to send over to you to make your job easier, essentially, but these all installed from OpenNeuro PY, so that was originally accessible data sets.  
and more columns out of which we take out 48 features for them because the other columns did not discuss the neurons of the brain activity that were relevant for an emotion. The original data set was not for emotions. So we take out the relevant columns that discuss emotions and the label emotions. And then we filter it out.  
to have a data set that has a label, that has a target label as emotions, and the parts of the bridge that are actually involved in emotions. So the reason I'm asking this question is this. So when the time comes to do the write-up.  
What I see is there's going to be a paragraph that says that's the CSV file that were the files that we used, the top of the tables that we used to generate these results are derived from, and then there would be a citation of these two or three data sets. So just to  
To repeat, one you said is called open neural. So that was where the data was downloaded from. So this is like a portal.  
Yeah, open Nguyen.  
So if we had to reference the exact data set, how difficult would that be? Not very, it's in GitHub. So, oh, it's on the GitHub. Perfect, okay. So, and just to be pedantic, I'm sorry, I'm pedantic. If we looked at the GitHub and said, where in the directory structure, can we find the reference to the data sets?  
This is just to make our lives easier when the time comes to write up, because then we can refer the AI to the transcription and say the place in the data set where you can find the sources where this data is.  
Well, if you bother an address there, Pritish.  
Yes, that's what I was looking at. I was trying to find out. Okay, so alright. So this is the pandas where...  
Where the data was downloaded from.  
OK, so open your download. So here's the so we have in the video understanding human brain emotion exports untitled 1-2 dot HTML. We have a copy of the notebook and then there is.  
line 8 where it says you open neural PI download data set DS 005700 target directory but it looks like the identifier here is this D for Delta S for Sierra 005700 is that fair?  
I believe so, yes. Okay, so that's what goes in the paper as the source of the data.  
which is important because, you know, referees, they'll say where that data come from and it's just stated up front, which makes their jobs easier.  
It neuro an FMRI. Oh, there it is. Neuro MO.  
OK, so my data set.  
That's not it. That is it. That is it. OK. So the data set we're looking at is exactly that. We've gone to open euro.org datasets and then we use that identifier. We have 005700 and we have some versions there. The data set is called.  
Neural MO on FMRI data set for motion recognition. Amazing will not publish.  
I believe it was 2025, yeah, 10 months ago, ohh, last update, 10 months ago, but it's a fresh data set.  
Wow, I'm excited. Sorry, Pritish.  
Carry on. So what happens next? Global geometric projection, centroid PCA. What's that telling us? Yes, so these are the kind of plots that we had already looked at. Yeah, we I plotted these embeddings in the 768 dimensions and then.  
good PCA to bring them down to two dimensions to be able to see them. We see kind of clouds being formation formed for every emotion. And after looking at the silhouette scores, it's clear that even though the pre-trained models did cluster them to some point from clouds for them and separated them one emotion from the other.  
But they were not proper clusters, so the way that I mean, think of them being clouds and not islands, because something like this can be strong, like I call it islands, because they're tightly packed in breed, and this is from the after the fine tuning anyways.  
But we looked at how they look there. This is something that we've already covered, but this time I also did it for the brain data set. What's interesting is that in the brain data set, there was not clear cloud formation.  
So, even though we got centroids of which were clearly apart from each other, but there was not, like, the silhouette score was negative. That means there wasn't... Yeah, go on, sorry. The silhouette score was negative. That means there wasn't a well in that.  
Defines, but I imagine the data is very sparse as well, comparatively, so maybe that can be said. Yes, that that is there, but it's still every emotion still has 40 data points, right? And a negative silhouette scores means that you took the original.  
results of the pre-trained model, you isolate the end stage embeddings into actually out of these embeddings, there's only really four or five or so.  
parts of that that are predictable that can predict.  
Remember, you showed me the graph that, yes, that has 40 different metrics, the brain.  
Right, the brain, there's 40 different metrics that is like, oh, these are activation. Are there any metrics that present more?  
Within.  
each centroid. Does that make sense? Does what I'm saying make sense? So for example, in the centroid for fear, it's not fear, it's disgust, right? In the centroid for disgust, is there...  
Imagine that it's the embedding, it's the brain embedding at the end stage of that brain embedding.  
Are there any are there any parts of that embedding that weight more towards the output? So, and that's what that is what that is one thing that is a good question you that you asked here, because that is the one thing that we also discussed. This is the information that we lose when we train these model when we.  
use that information as embedding. So initially this data set, you look at how they have names for all of the columns they have, it's the information about which, because let's say Angular Gyrus.  
and how activated it is for an emotion. Now, you can imagine that there has been an initial study where there are parts of brains that are activated during a particular emotion. And when we treat them as embeddings, we lose the column names. The model.  
Will not be able to separate this data point, this activation, and this state activation, right? Because let's say these, they are doing these two places, they're activated together when we're sad, but the model doesn't know that these two activation points are right next to each other.  
The location of the is is missing is lost when we turn them into embeddings, is there? So, just gonna add, even though there are no clear clusters visible, when you bring it down again to a statistical level, statistically, those...  
clusters are very much there and they are very much there above chance. I believe the p-value is like 0.001, which is highly significant, showing that there are clear clusters. They are just not visible when you bring everything down to a 2D image because of how the brain data is separated in comparison to the LLMs. So what does this contradicts our silhouette?  
School.  
negative. So what does the p-value tell?  
statistical significance. Yeah. What does that mean? Sorry. Okay, hold on. Let me, I've got another paragraph that explains this again.  
Although low dimensional visualizations of the brain, brain's representational space, do not exhibit clearly separated clusters, this does not imply an absence of structure. In A high dimensional space, so in your 48 dimensions, the emotional states form overlapping but statistically separable distributions.  
This is confirmed when we did the low so decoding, which achieves an accuracy of 0.56, substantially above the five class chance level, which is 0.20. And then with the 95% bootstrap confidence interval, we got 0.49 to 0.63.  
A permutation test further demonstrated that this performance is highly significant with P0.001, with real accuracy far exceeding the null distribution, which is your mean and SD. The apparent lack of distinct clusters and 2D projections arise from dimensionality reduction,  
and substantial inter-subject variability, which then blurs the boundaries when aggregated. Biologically, this reflects the distributed and multifunctional nature of neural systems where emotional states are encoded as graded overlapping patterns of network activity rather than discrete isolated categories.  
Well, I'm super impressed, but I don't understand a lot of what you said. But I think it's good because that's the nature of science, right? Everyone's going to understand everything and that's what you're bringing into the paper and you're the specialist, the statistical biomedical specialist, and that's your take and that's the interpretation. And if you're happy with that, I'm happy with that. But I don't understand it. I have to work a lot on it. So I have to.  
what you're saying? The statistics are confirming that is the brain mapping is structured, so there is a clear structure. It's not randomized by any means. We've got a clear structure of what is happening in the brain. They just cannot be discreetly clustered in like a 2D.  
Thanks, yeah, super. Oh, yeah, that'll make sense because that'll make sense later on in my experiment as well explaining it. So one more question, Amy, what you just read is, which is the second chunk you've read of something you've actually written, have we got that on GitHub?  
No, nothing. No, but I have a ton of Word documents where I like got loads of sections of writing separated, and I think I spoke to you when we come to compile everything, those can just always. I think there's a big push coming up in the GitHub repository where my experiments, her work.  
And then what we, I mean, we can do that first and then me and Josh were going to have to repeat whatever we did right before this to kind of like right validate everything that we've done so far with statistics.  
So there's one thing, so we paused at the moment you were saying that there is something that is lost when we put the 40 dimensions into a vector because it could be that the proximity of two activations is lost. Is there any way to preserve? Sorry, I was, I zoned out for two seconds.  
That's OK. Go on, Josh. No, I'm also wary of where are you in the in your explanation now? How far of the way through are you? One third, one third.  
We did a lot of work. Well, I was going to say because essentially the essence of what I was going to say is that yes, you lose location data, but location data doesn't necessarily exist in embedding space for the LLME. That's why it's a fair comparison. It is a fair comparison.  
Except, in your case, you isolated the main markers of this is in the embedding space. This is where it's at, because a neuron in neural system of the human brain, the location of the neuron also matters, although in the embedding space.  
I mean, there's no concept of location for them, and they they're all embedding space is the embedding space. The values in the embedding space contain location like directions, they they location as a it's implicit, but yes, they're in the last stage, but they have to have gone all through all this process.  
It is not for now, but separately, if, if you can reduce, if you can highlight, if you can do the same thing that you did with your with LLMs, when you check out features and only sort the job, I've done that for premium, you done that, okay?  
I was going to say, we did a lot of work to get this, so, so we do lose that, but to make this comparison fair, I mean, it's also true that I didn't think if there is a way to keep that into context, the location of that.  
But I couldn't think of anything. So as far as I know, with LLMs, you get the text. With the text, you create a dictionary. With the dictionary, you get numbers. The numbers are tokens. With the tokens, you generate an embedding. Because you have these numbers, you preserve sequence because those tokens fall.  
In kind of predicted, yes, no, I understand that that's not what happened here; we just said, no, we lost the numbers, like we have to have to add positional embeddings.  
So that's maybe future work that we said. Yes. Future work, we have to point out that these embeddings were taken as is, and to simulate exactly what an LLM is, we would have to have some kind of positional embedding reserved to perceive the structure of the whole brain.  
Where, if I'm saying this is 1 data, this is 1 data, I'm also telling the model that these two are closed points. Yeah, something like this is a further away. I think that's a good one for future work. It's the same as how, when we train the model, and we now train the VLM, you put in a second model. Remember what we were talking about last time? We had the second model.  
That second model for us is speech or whatever how it comes out. You can have that data because we can't have that data. But in separate space, we might have that data and that's what that is. Back in 5 minutes. If we go over any more biology stuff, just remember the question. So when I come back, OK, so.  
Yes, that's what we found that the clusters were much tighter for the fine-tuned models.  
properly placed and separated from other groups as compared to the clusters and pre-trained models.  
And this is confirmed with their silhouette scores as well. And then the silhouette scores is a thing for clusters. Yes. So the silhouette score, what's the deal? The closest, the lowest? So the Quinn native seems to have quite a high silhouette score compared to the other ones.  
So, what's that telling us? It's telling us that that higher silhouette score means that the clusters are tightly packed and well separated from other clusters. So, for instance, the B.G.E.D.S.A.I.R.I.  
So, the 2nd row there.  
Oh, they're separated. Brilliant. Perfect. I like it. So the number reflects what we're seeing there. Yes, the number reflects what we see here. So that's just the thing for the discussion to say, we observe that for that row that has a very low to do that score, the clusters are...  
Are spread out and for the not not distinct or tightly backed and distinguished from the others plus right? Cool, so that's a good thing for the discussion for that figure. So, this is the first test that I did, and...  
And...  
Yes, and then I'm sorry, I have to, I didn't understand this. I'll have to go back to, I lost.  
Please stupid.  
Okay, so now we're looking at phase two linear probing and 1D directionality.  
And that is from.  
The understanding text, understanding underscore text underscore embeddings folder again, forward slash reports, and the HTML page we're looking at is phase 21 score summary dot HTML.  
Okay, to be pushed to the GitHub.  
Or if it's not pushed, that's the reference on Pritish's MacBook. Let's go for it. What's happening here?  
Okay, so what I did here, that once we saw the clusters, whether it was these embeddings, 768 or 48, what we're saying that these embeddings hold the context of the emotions. Right, right. So we should be able to, if I can get a model.  
It should be able to learn to predict the emotion by just looking at one of the embeddings, right? Yeah, so what I did, I trained the regression model.  
So, on the that will that will take in all the embeddings and try to predict the emotions. Amazing, yes, because if if the embeddings has the context, then the model should be able to learn, predict the emotion right from the embeddings.  
And the accuracies are really good, besides the this this data set, which I found out was not was very noisy, and brain FMRI as well. Apparently, even though the the brain FMRI from those embeddings.  
We could not train above 90% accuracy model.  
Sorry, say that again, you could not train above 90%? Yes, the model that we trained on the brain data set was not able to reach over 90% accuracy.  
With the embeddings, the with the 48 dimensional embeddings, so if we OK.  
So, are we saving computation by training a?  
So, what's happening here?  
So imagine we're writing something in the paper and saying, well, one of the experiments that was done here was we took the embeddings, with the embeddings we generated, we trained the regression model. So now we have this pipeline where we can go all the way up to the embedding. That's the only thing we need, the embedding.  
And then from the embedding, we use this regression model, that's 95%. So have we got the angle here, like green AI, it's going to consume less electricity. So what's the? The angle is, the angle is that I wanted to check how well is the context of the emotion.  
Reserved in the embeddings, because if the if if the embedding has the proper context of which emotion that the text, then we should be able to predict the emotion of the text just from the embedding without looking at the text. Right.  
So...  
We go the embeddings, did we take it from the last layer? These are the same embeddings that that we used last time, that we used last time, which they are the last layer, get that set of embeddings and say, based on these embeddings, predict the emotion, yes, so it's not going to like output token outputs, no, no, no, no, etcetera.  
What I want to do is test, do the embeddings, how properly the context of the emotion was conserved in it, and which out of the 768 directions that these embeddings have, which directions hold more context and are more capable of predicting the emotions than the other direction.  
Because of those 768 dimensions, if the model is storing the model, when it then converts a text into 768 size embeddings, it's holding context of a lot of things and not just emotions, so...  
I'm assuming that only few of those directions are actually important for holding the emotions and not all 768 of them. Right, so some of the elements in this 768 vector are more important, yes, more than others. Yes.  
In context of emotions, right? So, where the quantitative and 768 are the pre-trained models? These are the fine-tuned models, and these are the pre-trained models, and what is the best score of the pre-trained model?  
Uh, 95.59 and the best is 97 on this, so the data leakage because your...  
What did you predict on? What did you predict based on the embeddings in the same data set? Yes, no, I imagine, imagine this is a transformer model. I understand how it works. I'm saying you predicted you used the embeddings from the original data set, but you've also trained on that data set.  
That's data leakage, no?  
How would that be? You trained on it and you're using it to predict? No, so I've inside those embeddings I created test and training set to test the model. So those two brain models are the ones that were fine-tuned on the ABC.  
Yes, wow, and they will on the test set specifically.  
I don't remember. You what? Were they were they fine-tuned on the test set or the whole? Sorry, I'm being pedantic, but it's fine-tuned on the test set. It stayed separate the whole time. Well, that's something I have to check. I don't know. I have to look at the experiment. No, no, you give me about embeddings of the validation set. So when we fine-tuned, sorry.  
Fine-tuned the model when model we took all of the data set, we divided into test, train and test, and we train the model on train, we took all the embeddings on test to look how did the how the test embeddings are in the 768 space, right? I think that's a good one for the discussion as well for the methodology.  
Yes, that the quen, both of those quens were, and then the methodology we described. Cool. Okay, and then, yes, then we got higher accuracy for the fine-tuned model, which was sort of expected, but I was also surprised looking at even pre-trained models.  
So by pre-trained models, they were not, they don't have the objective of predicting emotions. And these embeddings at 768 dimensions, we were able to see clusters from those embeddings, so we know that the emotion is somehow kept there. But I want to check how well they are then. And from those 768 dimensions, even when the model...  
was not given the objective of taking on the emotions. Even then, those 768 embeddings were able to predict at an accuracy of 96 or 95.29 and 95.59 percentage the emotions. That's, yes, that's right. So.  
and this one dimensional projection accuracy. So this test I did to check if I bring it down to 1 dimension, if I reduce the 768 to 1 dimension.  
Does it still hold the accuracy? And it kind of did.  
Wow.  
So, but, but this, but this one dimension is gonna, it's gonna point the same direction in space every time, no, most two negative and positive.  
Somewhere here by this emotion, this emotion, that right? Ohh, got you. I can cluster that way in one dimension. Wow, but but this test of mine, I need to do it with more dimensions because I want to see how many lower dimensions can we bring it down to, right, while still holding the context, but if you went all the way down to 1 dimension.  
That's crazy, and it's still here. I know, but I want to do it for 10, 15, and 20 to make sure that, but that could be for future work. Yes, well, that we say in future work, God, it's in that we could have more than the maximum number and one and have a few in between.  
Yeah, it can not be a good one. And then you have global interpretations. Is that what you just described? Like the bullet points there? Yes. Okay, super. So that, and just to confirm, the lower number with the brain MRI, is that, are we saying that's happening because the data set's not big?  
Alright.  
No, we don't know. You don't know. Okay, cool. I'm just saying that those 48 size embeddings, when we train a model to try to predict the emotions out of them, the model did not perform very good. But it's also true that there was only 200. Right, so that's something, but it's better than what you call it, much better than guessing. Yes, it's better than chance.  
5 emotions, so anything over 20%, that means it is able to hold some context. 63% is better than 20%. Oh yeah, that's not considered like highly significant because yeah, 20 is the board I'm sorry.  
So now we're looking at phase three signal decay and redundancy, and that is...  
So that's the same folder and the name of the HTML file is phase 3 underscore summary dot HTML. So yeah, take it away Pritish.  
So at this stage, now that we have modeled that is trained on those 768 embeddings to predict the emotions.  
From the model, we know we can determine how much weight does the model put on each of those dimensions separately to try to predict the emotions. Say that again. We can predict how much weight the model puts on each dimension separately from the trained model.  
Yeah. We know that how much weight was it putting on those dimensions separately to try to predict the model. How important, so if I put it in limit language, how important is one of those dimensions to get to emotion? How important is that one dimension?  
And we get it for all of them. So in essence, we can out of the, we can put, let's say, dimension one, two, three, 4, until 768. We can create a list sort to sort them out where we can say, okay, this particular dimension is the most important when predicting emotions and so on. We can get a list. So let's say we've.  
Get a list, you can you can list that in in in descending order, say this is the most important, yes, it's the second most important, wow, so essentially, yes, this is the experiment you're describing is this signal decay and redundancy, so imagine that.  
But imagine that the 768 dimension list, we get which ones are important, more important, and which ones are not. We list them out that way, right? And then what I did, well, I wanted to see if I, so basically, I found out that how...  
what are the important directions in those embeddings that actually contain more information about the emotions? Because again, not all of the 768 would contain information about the emotions. Only some of them will. By this experiment, we get that subset. Signal decay and redundancy.  
And then the signal decay is like the strongest signal is the one that's most related. Yes. Okay. So the redundancy, what's the idea that certain elements in the vector have the same signal? So we don't need both. What's that's not signal decay. This, the listing out of a priority list.  
That was done between phase two and phase three, and not I won't say between phase two and that was their final, because I trained the model in phase two, I have that list now, right? So I used that for phase three. Phase 3 experiment was that now that I have this list of which directions are important, I start taking out from the top.  
Take the five top out, trade it again. What do you get accuracy do you get now? Take another five out. What accuracy do you get now? So, in essence, as I'm taking the more important dimensions out, the model accuracy should drop, right? The model accuracy should go out, drop, yeah, because I'm taking drop.  
Always taking the important ones out, if you take the important ones out and train the model again, with the less important, the accuracy should drop. Yes, so that's what is the signal decay, yes, right? That's very important because that would fall in the category of ablation test. The referees are very hot on, and they say, well...  
Your system, it's kind of solid, it's explainable, but what happens if you try to break it? And that's exactly where you're trying to do this. Yes, I tried to break it. I found out that these ones are important, so I took them up. And then you say, and then the accuracy goes down as a result. It goes down. The test was done different. So what I found out.  
Was that, so what I would call this in the paper is ablation test and the ablation test and then the signal decay can go in the text. It's just I don't see this as a standard term in the what I'm gonna call the literature, but we it can be called signal decay. I know, and if it's called ablation, so...  
Because I knew what I was doing, but I've not done things like this before, so I wouldn't know the nomenclature, right? What about redundancy? What's what does the term redundancy explain in the experiment? I kept on training my name again and again after, so you you kept repeating, yes, training, remove the top.  
Train again, repeat, remove another five, repeat, another five, repeat. So what we're seeing here on the x-axis is the number of things you removed. Yes. These are the number of linear dimensions removed from the topmost priority ones. So the defines most important, the 10 to 15, the 20 to 20.  
Five, and then for these five guys here, so what do the three extreme, what do like 2 extreme cases show us here? Yes, now these dotted lines, they are chance. So, as by chance I mean that if if there's the there are five emotions, yeah, that the model is trying to predict, then the accuracy.  
of 0.2 or less is invalid. It is guessing, right? And for six emotions, it lies somewhere around one, somewhere between 00 to 0.2. The guessing, yes, for this thing. OK, so, and we have it here, so the chance with text is this one here, the lowest one that's below 0.2, it could be 0.1 or less.  
or a.15 and.2 is chance with a brain data.  
Okay, so the brain MRI is this guy here and it's there.  
So, we start on fifty-five percent, and then we drop to, and then we only have 40, so we only, we only take out forty-eight plus 12 of 12, and then we didn't wanna take less than 12 hours, and like this guy up here, when 7 sixty-eight, we start near.  
Of fine-tuned models, right? This is the same model, by the way. This is just PCA done on embeddings, right? So, if we take five, it's still very close. If we take 10, it drops; if we take 15, if it takes twenty-five, it says, "Bad as guessing." Let's see where...  
Yes, and then if you can retest that plateau space again, you know, when we go through the experiments, we can retest that plateau space. So, what happened here is that, so I'm loving this experiment because I've never heard of this. It's like wrangling with.  
The yes, the embeddings, so what's happened? That's explain what I, yeah, carry on, sorry. So imagine the 768 size embeddings right here, and in a pre-trained model, the context of emotions was spread out in the directions. That is why, even if we start taking out the top priority ones and accuracy did drop.  
but it was more linear drop. So even the ones that were down below, not that prior, not that top at the priority list would hold some context of the emotions. These models, they had some context throughout, but when you fine tune a model.  
Then, what happens is that now they're trying to keep the context in a smaller package, right, with more properly. So, in those, yes, more information and that, yes, and and it's amazing because even though the first what happened was because in a fine in a pre-trained model, the context about emotions was spread out in those directions, yeah.  
So, as you keep taking, as you keep taking them away, the accuracy drops linearly, but in the first five, because that's 768 and what's that 763? Yes, 763. That's so that's in the first five dimensions.  
Yes, they all do this all unnecessary costs.  
But, well, a lot of the first in the 1st 10, it's like, well, in the first five, it, yes, yeah, yeah, sorry, sorry, yes, in the first 10, it holds most of the content, the first five, right, then a fine-tune model holds a lot of that in the first five or six, so now I'm messing that it was kept in the first.  
Maybe 10, because 10, it dropped a lot, like, yes, 21 to 10, it's like dropped 30%. So, I mean, imagine the context of emotions was brought down to 10 directions, among about 10 directions, and very tiny feedback, yes. Ohh, do we stream 10 dimensions? You have to keep the context of the emotions.  
Well, it's the thought that you mentioned to see if that's moving, but this is I love this linear thing, 'cause it's more spread out, right? Yes, that's what you, that's what that's what you would think that in a pre-trained model, yeah, the context of emotion in all of the 768 dimensions was a little spread out, yeah, but when you fine-tune them.  
It kind of backs it, yes, and in a smaller subset, and and and it holds them very tightly, like, because even when you take all the top five, by the way, the accuracy did not drop a lot, so then imagine and imagining that maybe the fine-tuned model kept the context of emotion in top ten; I think top five, I would say.  
Because once you got past five and you removed more, no, no, even if I removed 5, it was still accurate. So even after five, there were more dimensions. The top five most among, I don't know how, what would be the term to call these dimensions, that the elements in the embedding that hold the most information about the emotion.  
Yeah, so what can we call those elements directions? It's embeddings, right? A signal? Yes, the strongest signal is in these top five. Once we no, not not in the top five, because even if we remove top five, the accuracy is high, right? Even after removal 5, so there are one or two. I'm imagining somewhere between 1:00 and 5:00.  
that still hold a lot of context about their emotions. But when we take on. 25 and 10, it starts dropping a lot. And if we go to like 15, well, we go to 20, it's like catastrophic. Yes, at 25, it's about to reach chance. So it's like that's 3% of the vector, right? Which means that in any case, you can look at the first five.  
How did you? Yeah, I see where you're going. We have a vector of five elements that captured that perfectly and said, well, we can actually hold this information in a vector. So maybe it was a describe. Do we know where these point to? Do we know where? So you've got.  
the information about the first five, you have positional information about, because you've ranked the, I have it, but I mean, I have to look at it because I was also like coding, right? So a lot of this was key. Okay, then once you find the top five priority, remove them. Yeah.  
We look at the weighted percentages that it's applying to all of these figures that you're removing. It is a model, so it is a NDY file. I would have to. It's what? It's like this kind of binary file. Yeah, it would have to be unpacked and looked at. We can, but I guess it's.  
Something that takes, well, I mean, it's not, it's not like company a file and say, where is it? It's like just, but it's not hard and it's, yeah, in the card, like you can remove aspects, then surely you can look at the weight that it's applying to set aspects to see, and if you can see pointers, you can see if there's...  
Let's say, imagine that I have a list of 768 numbers. I will know which one is more important. So yeah, so just to repeat your question, so it's on record here. So the idea would be identify how much, what's the weight of those numbers in this?  
When you train any model, like a machine learning model, you can look and actually look at how it applies the different weights to the different things to make its decisions. It'd be the same as this. It'd be looking, okay, we're taking this away and we're seeing a curve, the actual weight it is applying to those, but then you can get the actual percentages of  
you know, this is most important by X, Y, and Z, and that will give you a fuller figure of what exactly is happening with the numbers. Following what you just said, if we got this 7068 vector and removed one number and said, if you remove this one guy, what's the accuracy we get? And it would say, well, if we remove that one guy, the accuracy would be 99 point something, would be 99 point something less.  
And then we apply a weight and say, well, the weight must be, and if we, how would you see that working? If we remove 10 of those, we would see that the accuracy would drop. So would you apply a weight of those that are blocked and you would apply everything individually to every element of the vector?  
two elements in the vector. So unfortunately, I'm not the best at the mathematical side of things. I've done just trying to model how. Yeah, I mean, I've done things like this. In my brain, how it's working is like when you train a machine learning model and then you get that.  
output from the machine learning model, which gives you, for example, a curve very similar to this. Usually I need to feel like I need to pull up an old piece of work. You can then look at, okay, so we're dropping X amount. How are you making that decision to... The drop? Yeah.  
So, and then the weights should inform us why that. The weights would just be a mathematical calculation that is done based off the percentage that it's dropping, but I couldn't tell you that. Yeah, so, so let's think if we think about a multi-layer perceptron, it's got like it's all it's interconnected, then we have a number of weights there.  
We want to look at these weights separately and say, we know that the accuracy dropped at this point. What was the influence of each one of those numbers? We'd have the weights right there in front of us and we say, well, we have that the weight here was 0.2 and this was 0.1. So that value that came in here  
was the one that had the most influence on that draw. That's something in those lines, is that what we're trying to say here? Yeah, essentially along those lines, yes. So let's say we have a vector that represents an emotion here, and then we have an average, and we would want to say which elements of those vectors have the most weights in this draw.  
So we would have then a table next to the column that would say, well, these five guys here or these 10 guys here have the most weight in this draw.  
Right, so that's something to be explored. So, why this test is important to us, because we were talking about we want to validate a context in the embedding space.  
But we have the first thing that we need to do is figure out how many data context in the embedding space. So what's that mean?  
So what we're doing right, we're validating whether the 768 embedding space contains the context of the emotion correctly. So we're validating the emotional context to it, say, in this embedding space. And then for that, first thing we have to figure out were.  
how many directions actually hold the context of the emotion. And this graph tells us exactly that. So when I look, let's just look at the purple line, let's just look at one line. If we look at one line, then we know that the...  
There are like 4, there are five, somewhere between 5 and 10 directions out of the 768 dimensions that hold the most context. And when we go beyond 20 or 25, we've lost all context. So there are 20 directions in the 768 dimensions.  
that whole context of emotions. OK, so in vectors, so when we're talking about vectors, would you consider each element a direction? Yes, you would. OK, so because I'm not that familiar with vectors that I would call an element of direction. But if you're familiar and say, yeah, that's the way we describe this, I mean, you can call it signal or you can call it.  
So I thought that if we look at the embeddings, then we can say that all of the single element is a direction. OK. And the number itself is a magnitude in that direction. OK. So I'm trying to calculate how.  
or how many elements actually contain, how many directions or elements or signals contains the information about the option. So the way I would refer to it are elements, but if directions is used and describes it and someone who was reading it would say, okay, I understand the direction. Yes, in embedding space, you can call each of them as directions and then each one.  
Okay, and then a magnitude, and we can call a signal or magnitude, okay? And then, and so this also tells us what happens when we fine-tune a model on a context. It kind of compacts the context.  
In lesser dimensions, right, but more properly, however, in a pre-trained model, the context of the emotion was distributed, right? So, so those two lines there.  
Yeah, so the green line and the blue line that were the energy net balanced and the PG balanced, they decreased linearly.  
And then there is the BGDSAIRA, which we're kind of not considering really at the moment for some reason, but it's there in the plot.  
So I don't know what we have to point out about that, because if we had to point them out in groups when we're discussing this iterative decay curve, so we're saying, okay, we're performing an ablation test to see what we're seeing, and we've seen that the quit models behave like this, and then the PG and the MDF.  
NPNet model behaved in such a way, and the random right data behaved in such a way, what do we say about the orange? Orange line is similar to BGE and NPNet, right? It's just the data set wasn't clean, so I think we can remove it later.  
No, I understand. But I think it's good to leave it and just say that data set is noisier. Problem solved. We've discussed the results and we're not hiding noisy either. It's noisier and that's what we have. That's what my aim is to do for the next few days. That I clean things as statistics, I write statistics. I think the statistics are more important than clear. I won't bother about cleaning the data.  
I think this is a good result and we just saw in the discussion that data set is a bit easier. It's noisier. Perfect.  
And this, these are the numerics of it when we kept on removing dimensions, so when we removed, which reflects exactly what we've seen there, so that's exponential drop-offs, the two linear drops, and the noisy bits that is.  
Pretty much like the two linear bits.  
with a sample drop. Perfect. One thing that was worth mentioning here that in all of the LLMs, regardless of being fine-tuned or not pre-trained, when we remove the top few dimensions, at a point we have lost a lot of information. So even for the...  
pre-trained or fine-tuned ones. If we look at initial, which is 95.29, after 15 dimensions, 51, so lost 40% of the accuracy. Similarly, 47%, about 30 something or 40, here we lost 60 because we move back and then it's slow.  
Similarly, this way, but look at this brain FMRI only 23%.  
So, I mean, what I'm trying to say is that, because it was a lower dimensionality to start with, it was a better spread of information between those directions. Yes, it is that the concept of there, there are a few dimensions that hold a lot more context that exist, because the first four dimensions we lost about 20.  
Three percent of the accuracy, but after that, like, we're barely losing anything, right?  
So, let's just repeat that for the transcript for the discussion. So, to repeat in a frame in a better way, when LLM store context in the embedding space, they pack it in the poor priority directions, so the top directions in that priority list.  
As soon as you start taking them, the accuracy loss is made a lot. There's more accuracy loss. But however, in brain fMRI data set, the context of the emotions is widespread throughout the embeddings. And so even if we take the top few embeddings,  
On top, sorry, even if we take the top few priority directions from those embeddings, there's a lot of major loss, although it started with less accuracy, but the top is also not, but that was the third plot.  
So, what happened next?  
So, we're done with the which the things you wanted to show, and now she has the things that she wants to solve on the video. Anything, go for it. Yes, that you did. Do you want to share your screen or? Ohh, just like we did yesterday.  
Hold on your computer. Join the meeting.  
Mhm.  
The tests that were done yesterday were all ran against the LLM. There weren't any of the tests that I did independently on my laptop, so they're all on yours.  
But I don't, I thought I could. Anyways, Alex, no, no, let's locate this. It's important. So, Amy, do you remember like roughly what time it was when you did the tests? So let's try to find these files. It's been things around 9 P.m. yesterday. I think we opened up some of it. I just don't think we opened up all of it.  
This is a full summary I sent over to you, Pritish. Right. Essentially everything that we've done. Okay, so all the stuff is in that summary. Let's just make sure that we can reference files here and results.  
So the summary we're looking at, what's the name of that file?  
Uh, I'm going.  
World claims and when they both great stuff.  
So.  
You want to just bring up the name of this file, so we have a reference, a motion. Uh, it's, it should send me this on WhatsApp. This is, yeah, so yeah, so it saved as a motion, square geometry, and score 4, and square explorer, and square report, and score V.2.pdf.  
That is the PDF file we're looking at with Amy's results.  
So this was done yesterday, the 27th, I think, at 9 P.m. And the header is emotion geometry across brain and LLM systems. Would you be able to put that in your GitHub?  
The spice file, yeah. Okay, so can we, the main thing I'm interested in doing is to make sure we can map what's written here, the data files and URLs, so when the time comes to write up, we can link everything up.  
So if you could keep scrolling and then I'm just going to keep an eye open for what's in that last page. We do have a reference list down here.  
On the test, this is just all the tests that we've ran. OK, so it's important that we map these guys against tests.  
So, we have here in that table, result one, two, three, 4, 5. Can we map these roads to?  
So the methods, there's a geometry analysis, and then there's everything that follows from the methodology. And then there's, I imagine, a related work at some point. Then we have results one, two, three, four, and five. So there's here results too. We have low-so brain decoding, when high accuracy permutation and CI.  
Can we map that to anything in the file system? Yes, I'll just have to look at where I've kept those because what happened is that I shared a user files on the WhatsApp airdrop. Okay. And then I did not keep a track of because I thought that...  
No problem. So in the airdrop, there exists the files that we can map back to files in your file system. Is that right? Good. That's all we need to know. So then when we go back to your airdrop, we look at those files and then we are able to map them to those rows in the reports there.  
Yeah. Not fair. Okay, so I think we're good then. So do you want to explain your results or is it time to go or? No. I also sent you more statistics today. In terms of the results, I mean, you've been summarizing everything from the LLM perspective. I have a couple.  
Well, I wrote a couple links.  
Speech box areas, essentially, but rather than visually, or sorry, do you wanna explaining them all via speech? I can just send them, share your screen and go over some of the results. It's 9:00, by the way, right? So, if you guys wanna have a break, I would be interested in listening. If you wanna spend another half, I mean, I can give a brief of the tests that we did. I can explain that, and then...  
After that, if she gives us the exact numbers, then it makes sense. I mean, I have everything mapped out in this kind of way, yeah, because I knew that I was coming here and I knew that I was going to be expected to talk, so just the validation from very coding, instructed to talk, yeah, that big gradient findings, the divergence of global structure.  
redundancy and iterative decay, which he kind of went over, but I've just got it from a different perspective. And then the overall cross system, like insights of how it all kind of combines together. Yeah, I'm going to invite you to the meeting, and then if you want to join and share your screen. Yeah, sure.  
So, we are.  
So  
Re.  
Only here wrong game, what 28?  
Indian.  
Amy, what's your surname, Amy? Oh, even Amy is spelled weird, so I'm probably better to come out. So it's A-I-M-L-E.  
And then...  
Got along.  
I am sorry. Re AA.  
Ei.  
OK, so that's I'll send you that invite to make sure.  
You should get the invites, OK?  
It's coming for long. Deans. Yeah.  
It should be in our case, essentially, while she finds out, then we explain what what exactly will be basically human within LLM.  
So.  
In my email right now, I'm actually on Teams. Yes, we tested two different. We first thing that we tested was the density decay pattern that we looked at. That means when a data point moves further away, like how the density increases. So the testing, is that related to the MRI data to these?  
reducing the signal. No, no, no, no, reducing the signal, that thing that is different. Although I did that for brain MRI data as well, just to look, but then the other testing that I'm talking about relevant to what I did with Amy was comparing 768 embeddings of LLM with 48 embeddings of.  
The density. Sorry to interrupt. Oh, wait, I think I've got it. It wasn't letting me initially link to it because it said the meeting occurred in the past, but I've managed to sort it. Sorry. Yes, so first thing that we compared.  
was the density decay pattern that we see saw in LLF, where when we move away from centroid, the density or the number of points at the radial distance increases, and then it goes down. We compared that and how the statistics are, how similar these two patterns are for LLM space and...  
Very similar. They're similar. This works. That's shocking. Yeah, this is it. I that that result I have, so I can actually show it to you.  
Check the screen.  
So there are three dots. Oh, sorry, there's a screen thing. So you see how square these are patterns. We saw density going up and then down for LLM. We also saw it for brain data points where as we move away from centroid, the density increases and then it goes down. And the second thing we tested.  
Was.  
So how in LLM space we see that similar data, joy and love are closer, joy and fear are further away. We tested that for brain centroids as well.  
This one was a little surprising because, although logically you would assume that delighted and depressed should be further away, right? It's OK. We're delighted. What's happening here? How are you doing, sir? What time do you finish? Another half hour max and we're gone. Almost done.  
So, we will look up the thing now, 3:15. Well, we have out of hours. Give us another 15 minutes, boss. We're almost done. Is that OK? Fifteen minutes, yeah. OK. Thank you.  
Thank you.  
Yes, of course. I'm just gonna screen there. Yeah, I'm just gonna start by going over sharing screen. Yeah, it's not showing up there. I'll stop my. It should be a new team. There should be a it's something we can expand that.  
It's going for me to expand it.  
It is on, screen is on, right? It says I'm sharing my screen, not showing up here, unless am I not in the right machine?  
We leave the meeting and then maybe rejoin.  
You're not in this meeting. Well, alternatively, if you send an email of your thing to either me or Pritish and we can scan it. I'll try to add you to the call. Do you get a notification on here? There we go. That's perfect.  
You can change.

PG-Bottrill-Frost, Aimee** 1:37:23  
OK, mute yourself.

PG-Verma, Pritish Ranjan** 1:37:26  
Measures.  
Done.  
That's success. Ohh, just kidding, just kidding.  
Ei.  
Okay, so Pritish has all the images of the results on his, but I can just go over quickly some of the, what I've gone through. I'm going to start with the brain decoding and validation, which essentially just goes over the fact of our data being, the brain data being efficient for this.  
So we run through a couple of different tests with the results showing that the brain representation does contain meaningful information. We used low source, so leave on subjects out, which achieved accuracy of 0.56 with chance of 0.20. 95% confidence intervals of 0.49 to 0.5.  
And importantly, the permutation test, which was highly significant, is 001. So this aligns with a large body of fMRI decoding literature, particularly, oh, I've got references in here and everything, sorry. You have references. In here, yeah.  
Sorry, following the work of James V. Haxby, which showed the cognitive and emotional states are encoded in distributed patterns rather than in single regions. So in a sense, our result is consistent with what's established in neuroscience. However, obviously, we're going to go and run through some things that haven't been previously.  
looked at in the area. So our results when it comes to the high variability and overlapping representation. So this just goes over again while we didn't observe the clustering physically, but how we got the statistical significant figures which show brain data.  
does have clusters and does have the patterns which make it comparable to this.  
So, despite significant decoding performance, the brain data does not form clearly separable clusters in low-dimensional visualizations. This is supported quantitatively by substantial variability across subjects with a mean accuracy of 0.56 and a standard deviation of 0.2.  
ranging from chance level performance to perfect classification. This strongly aligns again with existing neuroscience. Brain regions such as the insular and prefrontal cortex are known to participate in multiple of the processes, which leads to that overlapping activation pattern.  
What is novel here is the explicit demonstration that this variability does not imply a lack of structure. Instead, we show that the emotional states form overlapping, but statistically separable distributions in that high dimensional space. The implication that the brain encodes emotion in a flexible and graded manner rather than in discrete categories.  
explains why the visual clustering fails, but the decoding remains significant.  
Moving on to the ambiguity gradient, which is one of the findings that we've spoken about. It's what's observed in LLM space. Yes. These things kind of overlap and there's not the screens. Exactly. Yeah. Which is why there's such an interesting comparison between the two.  
So one of the central contribution findings is what looks like being the ambiguity gradient. We find that the representations move away from their emotional prototypes and classification uncertainty increases in a highly structured way. When we compare this relationship between the brain and LLM representations,  
we observe an extremely strong alignment with correlation approximately 0.56 with a permutation p-value the order of 10 to the minus 54, which again is practically 0 chance of it not being correlated.  
This result connects to prior work in representational similarity and prototype theory, but goes far beyond anything that's been studied yet. While earlier studies have examined whether representations are similar, they have not characterized how uncertainty behaves as a function of representational distance. So have you got references?  
that backup this state with the paragraphs just read. So I have about a million different things open on my computer still and stored away, which obviously I'll send over as a big lump sum of references for different. So we can, let's say if we have to fish out two or three reference, because if I'm not mistaken, this is the format for this conference is 1 page of references.  
It could be that we need to aggregate as much as the biomedical side as possible. It may be 5 to 10 references and so forth. Okay. Yeah, that'll be fine. I've already got more than five to 10 different things open. Sorry. What I'm seeing is that the issue is going to be reduced then to a set that's significantly enough represents all the claims that are being made.  
Well, there's not much studying that's been done and far of, like from a neuroscience perspective when looking at the geometry, because previously, essentially previously to having something to compare it to, this mapping structure that you're seeing, you can't relate it to anything. You can't say that it's significant in any way, shape or form. It's just a mapping structure that's occurring.  
Which is where most people have looked at, like balance and arousal being the two separate categories, because they tend to separate. OK, we're going to have a rebuilding, yeah, yeah, but thanks, go back over this, we've got all this here anyway on speech three, yeah, yeah.  
I'm sorry.

Sikar, Daniel** stopped transcription
