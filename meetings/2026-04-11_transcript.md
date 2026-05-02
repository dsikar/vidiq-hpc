**Meeting to transcribe-20260411_193505-Meeting Transcript**

11 April 2026, 6:35pm

1h 59m 12s

Sikar, Daniel** started transcription

PG-Verma, Pritish Ranjan** 0:03  
Close that, so let's just make sure it's transcribing.  
Hello, is it transcribing? Yes, it is, so we're good to go. OK, so we can, yeah, we can start from so last time when we met, I was gonna meet the...  
Okay, I think we need your mic. Let's see.  
Do we need the mic? Yes, we do.  
Right, OK.  
Hello.  
Yeah, yes, as soon as I turn it on.  
Right. So you were saying, so do you want to recap from last week and then we can carry on the density, the 10% drop and everything? Yes. So what we tried to do is we tried to plot these data points for every emotion and we tried to see, we tried to see the density of how.  
They are around its centroid. Right. I'm just going to describe this is just for the AI that's transcribing what we're looking at. So we're looking at the 2D graph. So we flattened it because we're looking at the embedding space. There's 768 dimensions. They've been flattened to two.  
And we're looking at what would be a, what's the name of that plot that only has dots, I forget.  
scatter plot. So we're looking at a scatter plot that's showing the density and we're looking at densities. So what Pritish has done here is he drew the dots representing sentiment and then radiuses representing density.  
Um...  
different quantities of density. Is that a fair representation of what we're looking at? So what we're looking at is I first I first looked at how the density is around the centroid for every emotion and every single emotion. So I'll just add.  
this is for the transcription and for the AI, that when we're talking about centroids, we've taken the average value of the embeddings for every sentiment. So that would correspond to a coordinate in this 768 dimensional system.  
and the average of all those values for each sentiment would then give us a centroid. So that's what we mean by centroid in this case. Okay, carry on. Okay, so...  
We also prepared a density decay map, which will just, and I had one more thing about the previous slide. So, if you if you go back one, so the sentiments we're looking at in this case are two specifically, it's joy and anger, and then we also we have an overlap.  
point that's denoted by a circumference. And then I'm just reading this up. We have an X in blue for the joy centroid. We have an X in orange for the anger centroid. We have a solid blue line for the joy belt.  
peak, which is where the density for joy is highest, and we have an orange solid line for the anger belt peak. Same case as explained before. So yeah, carry on. Yes, so our first assumption was that we thought that these...  
dots or these data points will be most tense around the centroid. That was the exemption before the experiment. Yes. But because we thought that most angry or most joy or any like would lie near the centroid. But when we tested, when we plotted these points out.  
We found out that the density grows as we go further away from the centroid. It peaks to a point and then starts gradually coming down as well. So we look at the next slide you had here, so we can describe that to the, so it's in the, so yeah, can you describe what we're seeing here? I'm not, but I'm not.  
presenting my screen. Okay, no worries. I know we just we're just presenting what we're what we're seeing. Yes, so we were discussing this. The density, yes. Can you describe what we're looking at now? Yes, so I'm looking at a line graph that in one of the axes.  
It represents the distance from the centroid, the X-axis, and the X-axis, and the Y-axis represents the density of these data points at that per unit, per unit.  
So can we go back to the graph? Yeah, so the distance from the centroid that we're seeing on the x-axis is an absolute, not a normalised distance. So we have a distance of 0 to 11 with no units. So that's, and then on the y-axis, we have a density per unit.  
So for instance, the peak for joy density is approximately at distance from centroid 9. And at distance from centroid 9, the value we have in density per unit is 350. So I'm imagining that the unit here is a unit of volume.  
Am I interpreting this correctly? Yes. Okay, so we're looking at a unit of volume in a 768 dimensional space. And the value that for the density from centroid for anger density is peaking at approximately 9 as well. And  
and then both begin to decay at approximately between 9 and 9.5. So they have thus they present the same characteristics as far as density distance from centroid and density per unit. And Pritish has noticed in a previous discussion,  
that every sentiment presents the same characteristics for this increase and drop in density. Is that fair? Yes, that is fair. And then if we look at other emotions and their density maps, then we find out that not just the pattern is similar, but  
all of them somehow seem to have their peak reaching somewhere between 8.5 to 10. So for in our case, the one that we were looking at was joy and anger, and we saw that the density starts to decaying between 9 and 9.5. But all the emotions have.  
a similar arrange somewhere between 8.5 and 10.5 where it happens for all of this, all of the emotions. So the, so we're looking at this 768 dimensional space and then for every example that we give to our LLM.  
The LLM gives us an answer. We look at the embeddings for that answer. So this is what the graph represents, is the average density for all the outputs. So the answers from the LLM and those in  
the embeddings representing those answers in the 768 dimensional space flattened into 2D so we can actually see in on a 2D graph. Can you talk a little bit about the dimensionality reduction techniques you use such that this plot could be obtained?  
Yes, so the dimensional reduction, let me let me go find it. We use PCA, which is one of them.  
Just one thing. So we're about to discuss dimensionality reduction. I need to go to the toilet. So I'll leave the transcription running because we're not going to transcribe anything. And I'll be back in 5 minutes. OK. OK. And if you want to eat anything else, Matt, help self. If you want to make a cup of tea, help self.  
All good. OK. And if you need to go to the toilet, just click down here on the right. OK. OK. Thank you. Thank you.  
I'll turn, I've turned on the mic now. OK, so, so we should be transcribing again. See, I think we are.  
Lee.  
Yes, alright.  
Okay, so for diamond, yeah, yes, reduction, we use PCA. Yeah, that's how we're reducing the dimensions for that.  
Rights.  
And with the PCA, we bring down the embeddings of 768 dimensions to two dimensions, which in our graph are named as PC1 and PC2, and the and we use we use those these two points as coordinates to plot them on the scatter plot.  
On the 2D, on the 2D, yes, but the the calculation of the centroid is done originally at 700, 700 sixty-eight dimensions, and then the PCA is done on that, right?  
Okay, so what are the things? Yes, the one thing that we noticed was how the density grows to one point and then it gradually comes down. Let's, and the peak in the scatter plot, we show the...  
peak of the density with the bold blue line for anger and bold orange line for fear. And what we are trying to show is how these overlaps.  
Generally, not generally, these overlaps almost always start right after the density peak of a class. Right, so I just want to describe what we're looking at again. So we're looking at the plot of anger and fear. So we talked about the classes before. So we're looking at.  
two classes only out of, I think, the seven classes. Five classes. Five classes we're looking at.  
So the title of the plot is Anger versus Fear Scatter, and as Pritish mentioned, the x-axis is PC1, the y-axis is PC2. The values on the x-axis go from minus 6 to 4, and the value on the  
The values on the x-axis go from minus 6 to 4. The values on the y-axis for BC2 are the same, minus 4. Oh, no, they go to minus 4 to 6. So that's the X&Y axis. What we're looking at is the  
The lines are.  
Rounded, so it's like this: a section of a circle, and then we see the orange, the solid line we described, and then the hashed line, this interrupted orange line, and we see the same for the blue, such that there's an intersection between the two.  
So imagine we're looking at the centroid. We've drawn 2 circumferences from the centroid, but we're only showing a section of it. So if we split 360 by 4, we're showing about 120 degrees maybe of the circumference. And we're looking at the circumference  
From the fear centroid, which is to the right, the arc goes to the left of the centroid. And then if we look at the anger centroid to the left, the circumference section goes to its right and there's an intersection between  
the orange, the solid orange, and the solid blue lines. And there's also an intersection between the hatched or the interrupted orange and blue lines. And what Pritish is describing is that the highest density would be up to the  
Solid lines.  
And then when the overlap starts happening, it would be after the solid line and up to the hatch line. Is that it? You mentioned there's a what happens like between the solid and the orange again? Yes. So the solid and the dotted line is just for a reference for me to cheque roughly how far is the  
Point where, after that, we're assuming that once the density reaches 10% of its peak density, that after those 10%, the data points might be outliers, right? Okay, so it's just as an assumption to test that.  
drawn the dotted lines. We don't have to treat it as something that's important in this visualisation in the sense that it's telling us something right away. For now, the only bold lines are the ones that we're majorly looking at for our experiment, which is the point where the density reaches its peak. Right. The dotted lines is just for my reference. I wanted to see if...  
if we can consider after the dotted lines as outliers, but that is not something that I've considered for sure. Okay, so the metric from a perspective of measuring descent dimensional space we're looking at, what's the significance of the distance between the dotted and the solid line?  
So you decided to plot to say, I'm going to plot that, that interrupted line. So you see this? Yeah. What I tried. So this would be the dotted, the bold line. Yeah. And dotted line would be somewhere around here with a dense. I'm just assuming that this might mean that it's on place. No problem. So what we're looking at.  
then is we're looking at the anger versus fear density decay, which is the one we described where the decay starts happening around distance from centroid 9. So what Pritish is saying.  
is that the hatched line, which is the interrupted line in this graph would be in this plot, would be approximately around 10 to be confirmed, but you said it's around there. So the graph we're looking at, the solid line, both for  
The anger and fear density would be around 9, give or take. And then the hatched line would be plotted at distance from centroid around 10. So that's what we're looking at in terms of metric. Okay.  
And then the third graph that we're looking at. Oh, sorry. Before we start the third graph, can we go back to the first one? Okay, so this one, in terms of dimensionality reduction, we're looking at PCI one. So in terms of dimensions here, PCI one and PCI 2,  
are something, a metric that was obtained by PCA. On those embeddings. On those embeddings. So this was the, these magnitudes or these quantities we're looking at here that go from minus 6 to 4 for PC1 and minus 4 to 6 and PC2.  
We obtained that through PCA dimensionality reduction. And what we're looking at in the following graph that shows the decay is an actual is that an actual quantity in the 768 dimensional space to say that the distance from the centroid is 9.  
when the density peaks and okay. Another thing that's all.  
Yeah, keep on.  
You need batteries for your phone for my laptop.  
Sorry, do you charge up? OK.  
Mm.  
I want to hear if you need it.  
Abraham.  
It's.  
Muir.  
The third.  
The third visualisation that we use for our experiment is the one where we...  
Read the regions from one certain distance. Just one thing. Can you describe what we're looking at here? This is good practise for presentation, because you're like, imagine you're presenting your MSC or a paper to an audience, and now there's a slide and you're presenting and there's a transcription. So imagine I'm the audience and you're telling me about this slide. So do you want to take it from this? Yes.  
Yes, so I think we have covered the two visualizations. Yes, I'll cover the third one. Yeah, this one, yeah, carry on. Yes. So the third visualisation you see is a bar graph and in which one of the axis, the x-axis is the distance from the centre that we have already discussed before from the centroids of.  
and emotion and the second axis, Y axis has the overlap count. So how many data points from the original data set overlap and lie in the other region? I mean other emotion. So.  
From our first.  
Oh, yes, the orange colour in the graph top is for fear overlap and the blue colour is for the anger overlap. There's only one thing I would mention here. I would start by saying what the title of the plot is. The plot we're looking at is the...  
The title of the plot is Anger versus Fear Bin Overlap Counts. Right.  
Now, when I say blue, is anger overlap? I mean, how many data points originally from anger class are now lying in the region that is that should be in the fear?  
So anger overlaps means how many data points from anger class in the embedding section lie in the fear section, fear region of embeddings. Given a radius. Given a radius. Okay. So, and then the radius we're looking at is here, right? Distance from center.  
Okay, yeah, carry on, carry on explaining, please.  
Yes, so what I've done is I've treated the I've treated the distances as bins, so from one certain distance from the centroid of a class to another certain distance at the centroid of that class, and in those bins, how many overlap exist?  
Okay, gotcha. Yeah, carry on.  
And then what we see is that these overlaps peak around a certain pin.  
Region, let's just look at something that's...  
that these overlaps start to peak around a certain region of distance from the centroid and this region is almost always right after the density peak.  
So, if you go back to the density, a peak plot.  
So we're back at a density peak plot. And what we're seeing is that if we compare now the distance from the centroid and take, for instance, distance 10, which is there, distance 10 from centroid, we can see that the density now has dropped.  
to 150 per unit, whereas it peaked at 350. So where we get 150, 10%. Actually, I'm treating the dotted line as 10%, so it should be around here, 35. Density reaches 10% of its peak density.  
You're treating, so this is important, that the dotted line you're treating at about 50%, 10% of the peak density. So here we see that the peak density is around 350. So that means 10% when the line would have been somewhere here. Gotcha. That's important. So can you say that again?  
Yes. Oh, and explain that on this graph. Okay, so we're looking at the graph of density decay for love and anger. And we see that in both of these graphs have a bell shape where they both reach their peak, anger reaches its peak density at closer to 9 point  
one or two and love reaches to its peak density at a region closer to 9.3. I just would edit the statement to say that it's not exactly a bell shape, it's more like a mountain or shark tune. I think that might be because of the number of data points I'm looking at it might, I mean, you know, if I...  
start noticing it taking more data points than it. Oh, I see. It would be bell shaped. Yes. So maybe this is just a graph on account that he can't represent all the data points. All right, fair enough. So it could, yeah, it's possibly it's bell shaped. I'll try to, but.  
So now we go back to the, so let's say it's bell-shaped. So now we go back to the hatchet line or the interrupted line. And what we're saying, can we go back to the density plot? So we're looking at this bell-shaped plot, which is the love versus anger density decay. And what Pritish said here, that on the y-axis, we can see  
that the density is peaking at 350 points per unit, and that that is at a distance from a centroid of about between 9 and 9.5. Now, what Pritish said that's important for us to add here is that the hatchet line, which is the interrupted line,  
If falls at under 50 dense points per unit, so at a density per unit Y scale, it would be below 50. And if we then run that line across parallel to the x-axis, we could see that that would be around  
10 units of distance, give or take, from the centroid. Okay, super.  
So now if we look at the amount of overlap, we'll see that the most overlap is happening, in fact, in a very low density area. And then we have a density count here, which is high, but it's just keeping in mind that although the density, which would be expected, that the overlap count is high because that's where the most overlap is.  
happens, we can also see that if we go back to, I mean, sorry, sorry, at that bar graph, the one we were looking at, we see that the overlap counts, there are no, oh, how interesting, there are no anger points here or they're overlapping.  
How's this?  
So we have no orange bar. Yes, but that might just be a...  
I mean, it could be that there are no orange points here at this distance, which is not a problem.  
But it's strange how it's similar for all of them.  
Anyways, that can be verified later, but generally what the pattern that we see is right after.  
The peak, we see that the overlap also peak, and it peaks somewhere closer, somewhere between the region of peak and diminishing of the density. So the overlap peak is in areas of low density. What's the size of the data sets?  
Thousands, 10s of thousands, hundreds of thousands. No, about 10s, about, let me, let me check. I think some of the images were less. I should have, I should have remembered. No, it's alright. It doesn't matter. We can carry on. That's one thing that we need for the paper, so.  
Yes.  
And, no, I don't have it right now. No worries.  
Anyways, so, and that's what we found out until last week, that the density reaches its peak and it goes down, and the region where it's going down after its peak is where the overlap hits the maximum.  
Nguyen.  
You know one thing that I think would be interesting here? I'm going to mention it now.  
This is something that.  
I do remember Sophia Mirvoda, she did a data visualisation workshop. So one thing she mentioned, the bar charts.  
that has a total and a...  
So let's say we're looking at anger and sadness and surprise. So she mentioned something like the sadness component is here.  
The surprise component, let's say there's more surprise than...  
then sadness is there. And then the total count is the distance of the bar. And then I was thinking something to keep in mind for future reference that would give us a chart that said, at this point, the total number of  
sadness is this much, the total number of joy is this much, and the total number between the two is at this height here. So we get an idea of what the total quantity is, and then the component quantities of that total. That's actually good.  
Just, and I'll remind you when, like, further along, but that's something she mentioned, like in that workshop, and now I'm seeing a practical unit breakthrough. Yes, I think we'll improve a lot of these visualisations before actual people. Yeah. Right now, this is just for us to see what's happening. And we could have a conversation with her as well. Say, this is the data, this we want to show, what you think we could use here. Yes.  
and hopefully still give us some good feedback. Okay, so from this one, so one last final experiment that I think I should show and then we can move on to our, so what we get to know from the experiments.  
I created a cluster for...  
all the up from the balanced data set. Before this, we were just looking at 2 emotions at a time. But I also created these clusters for all of the emotions at once to see how the centroids did. One.  
Can you comment on what you just said about the data set being unbalanced? Would you be able to have a discuss that just so we have it in the transcript? Okay, so initially in the data set, I found out that the classes were imbalanced. So in some of the emotions, like...  
anger and joy had a lot of data points, but for surprise, it was very less. So for running out further experiments from there, I balanced the data set in which I selected at randomly, I picked the least number of data points for a class, which was for surprise.  
And then I balance the data set by randomly selecting that many numbers from each of the classes. I do a random selection over the classes to ensure that I'm not...  
Taking away a certain, so you did a random selection to ensure you were selecting the same numbers for each crosses. Was this for this plot here or in general? I've even for even no before this for the two emotions I did the same thing, so even for the two emotions experiment.  
We have done it where the numbers are the same. Same. OK. So that's something we need to look at in terms of training.  
So how are we going to, because ideally we wouldn't want to remove data, we'd want to add data. And then the next question is, well, how do we add more surprise? And then that's a discussion to be had at some point. Okay.  
So now you're talking about the plot where you plotted all the emotions in a single plot and you balanced it to make sure that all the emotions were equally represented in the plot. Yes. And then I plotted separate clusters for every emotion.  
This was to verify our peak density and 10% density lines. Can you describe what we're looking at here, please? Yes. As if you were presenting it to an audience, like about the paper. So we're looking at a graph that the title of the graph is Surprise Snapshot. And it's a cluster.  
It's a.  
scatter plot for emotion surprise.  
The X&Y axis are PCA dimension one and PCA dimension 2, which were previously mentioned as the two dimensions that we bring down.  
the embeddings of 768 dimensions too. And then they're plotted on. You might want to mention this, the range. The range for PCA dimension one is minus 6 to 6, and the range for PCA dimension 2 is minus.  
roughly minus 5 to 5.  
And then in the and in the scatter plot, where we're locating the centroid of the scatter, and then the peak density is shown with the dotted black line, and then the 10th or the region where the density reaches 10% of its peak.  
density is shown by dotted red lines.  
And then we do the same thing for every emotion.  
I just wanted to add, if you go back to...  
that we have a legend. The legend reads others. It's like a grey solid ball, surprise, or circle, surprise, which is a blue solid circle, and surprise centroid, which is a yellow star. So we see a yellow star as a centroid. It's labelled surprise.  
And then we see the scatter plot blue for surprise points and grey for others.  
Then after this, we also have another plot, which is all clusters, balanced data set, all clusters in this graph. Yeah, carry on. The X dimension is again PCA dimension one and Y is PCA dimension 2. And we're trying to show all of  
the emotions in one plot. The range for PCA dimension one is minus 6 to 6, and the range for PCA dimension 2 is minus 5 to 5. The legend reads that the blue dots represents sadness data points. This orange dot represents joy, green dots represents love.  
Red dots represent anger, purple dots represent fear, and there is a brown dot that represents surprise. And with that, we have also used the same yellow star method to show the centroids of every emotion. And right next to the stars, we have labelled that which...  
emotion that that centroids belong to. Super, very good.  
Okay, so these are all the... So what can we say about the plots?  
Yes, so from the plot I found out that now we come to the findings.  
From this graph, we found out how the that contextually similar.  
Classes usually remain closer in this dimensional space than emotions that are contextually opposite.  
So.  
Something like joy and love, the centroids of these emotions are very close and they are, and both of these emotions are very far from centroids like anger and sadness.  
And then...  
Sam.  
surprise and then the surprise and fear lies somewhere in the middle of these two centroids and surprise being more of which is can be associated with emotions like a positive emotions more is closer to joy and love.  
and fear, which can be inclined towards a negative emotion. So it's closer to other negative emotions like sadness and anger.  
But you have an interesting insight as well. That surprise can be seen not as an emotion, but as a reaction. Yes. Was that the line? Surprise is more of a reaction than an emotion. And what we see in the plot here is a surprise lies kind of midway between love, joy, anger, and hate.  
And then the next one that comes along to the right is, what's that one, fear? Fear. But we can see the fear, sadness and anger are close. Yes, because they are. Yeah, because they're kind of negative. I'm going to call it negative emotional spectrum. And we can see that love and joy are also  
very close and that there's this separation between love, joy, and then fear, anger, sadness. And we can see those clusters are clearly separated and the surprise lies somewhere in the middle. And then we had the discussion.  
We have come down to two dimensions. We are assuming that surprise is somewhere in the middle. However, in the 768 dimensions, it is unclear whether this surprise would actually be somewhere in the middle. It might just be far away. But when we look at it from a two-dimensional angle, we think that surprise comes in the middle of it.  
I would suggest a comment.  
We don't understand, at least I don't understand very well, high dimensional spaces and how things behave in these in a high dimensional space, but my intuition would be...  
If we were to measure distances from centroids and then went to the 768 dimensional space without looking at PCA, just like Euclidean distance, which we can do in any dimension between.  
If we now start measuring, what's the Euclidean distance? And we could have a table here that said, what's the Euclidean distance between love and joy? And it would say the Euclidean distance between the two is whatever it is. Let's say it's one unit of distance between the two. And then we'd say, well,  
What is the Euclidean distance? This is being transcribed, by the way, but yeah, take notes all good. If you, what's the Euclidean distance? Is it being transcribed? Yes, we are. I'm writing this as a next set of experiments.  
What's the Euclidean distance between love?  
and joy and surprise. And we had expect, well, it's going to say something, and let's get, let's just suppose it's going to be two units of Euclidean distance between the two. And then we say, well, what's the Euclidean distance between fear, anger, and sadness? And then it would say, well, the Euclidean distance between fear and anger is 1.  
the Euclidean distance between fear and joy and sadness is 2, and the Euclidean distance between anger and sadness is 1. So we'd say, okay, so fear is closer to anger than it is to sadness, and anger is closer to  
sadness then. So we would be able to say that kind of thing. Then the next question we would say is, what's the distance between anger to surprise? And it would say, well, it's about 3 units of Euclidean distance. What's the distance between anger and joy? And it would say, well, it's about 5 units of Euclidean distance. What's the distance between  
anger and love, it's about 5 units. So we don't know exactly where these points are. It could, but we can see the distance, but we can see that there are distances that show that that's how those centroids are separated from each other. Yes, absolutely. That's that would be the right term. That would give us.  
a more clear idea of how these two centroids are, how close these two centroids are in the 768 dimensions and not actually right after PC. Yes, the PCA is just to visualize, right? Yes, yes, it should not be to assume.  
It should just be to visualize. Suggestion, just one more thing. The suggestion for the paper that I did, this is, I think, a couple of sessions back, or it might have been during the hackathon, is that we get this plot, we're looking at the balanced data set, all clusters, and turn this into...  
possibly an 8 or a 10 plot, one right next to the other.  
the fit as a full plot on a page. So we would have like the full, the page, the whole width of the page, and then have four or five plots. And on each plot, we would have only two classes that were plotting. So in this plot, we're only plotting.  
surprise and anger. And we see two, what you call it, a scatterpot with two classes. And then on the neighbouring one, we say on this one, we're only plotting anger and joy. So we would see like 2 by two.  
That's the idea. Okay. And then we'd have one maybe at the end that says this is everything with perhaps the only the centroids and then the points maybe with a transparency set high so we can see that there are points behind it and say, okay, that's the general kind of layout.  
And then, yeah, that kind of thing, but with two, two of the emotions, with two by two, yeah, saying, oh, now we're comparing anger and love; they're the two clouds, that kind of thing.  
Yes, so now this, this is what we see, this is what we can listen, let me see, get conclude from these graphs that these centroids are closer, but we need to also discuss the thing that we were discussing back.  
Then, how centre is a band order?  
So, the density, yes, so let me go back to bank, so that what we found that was counterintuitive here.  
is that the highest density is not exactly where the centroid is. And the further we move away from the centroid up to a point, the density decreases. So for instance, can you describe what we're looking at here? Yes, so for instance, let me choose a...  
Yes, so for instance, we are looking at a density decay graph between love and anger. The title is Love versus Anger Density DDK. The legend shows two different lines. Blue lines is for love density and orange line is for anger density. At x-axis, we have the...  
Distance from the centroid, which ranges from 8 to 10.5, and on the y-axis we have density per unit volume, which goes from 50 to 350. We see that both of these densities.  
be at around 350 and at roughly similar distance between 9 and 9.5. So the counterintuitive.  
finding from this was that when we, before the experiment, we assumed that the density will be peak at near centroid, but that was not the case. The density actually peaks at a certain distance away from the centroid. Now this makes  
Us to believe that the most angry.  
Almost anger, which the centre might not be the centroid itself, it might be the belt, and this belt around the centre contains them.  
Most, so if we're looking at this scatter plot for...  
Okay.  
angle then and we have a centroid at a point, we can say that the most density is actually around a certain region away from the centroid, which in this instance is somewhere between 9 and 9.5. So we can say that the scatter plot for  
anger gets the most tense around 9.5 unit distance away from the centroid. Now this makes us, this makes me also think about how a cent when we're talking about these sentences.  
It's anger is just one of the direction in the embeddings. And so these sentences will, although yes, have mostly anger, but also have probabilities for every other classes, no matter how small these values, probabilities are. So we can assume that these.  
A...  
The embeddings being away from the centroid at different directions to be, how do I how do I put it right? So I'll try to explain it to you what I'm trying to say. I think that these belt regions, yes, this is this is an important part what I thought about. Yeah, let's consider that this is the centre and we have a belt region around where the density is peaking. So we're saying that this is the region where the density.  
Whether there is mostly most anger, but we also know that there's a point, let's say, here over for love, there's a point here for hate, some other emotion. Right now, I'm just trying to vaguely talk about it, and we might drift away from the class. So, what I'm trying to say is that these points...  
We also have you when I'm angry, I can also I can we can associate this anger with because of anger from love. When I'm anger, maybe out of hate, maybe out of fear. So these, the one sentence saying that it represents anger doesn't say that it only has anger.  
So maybe something that the anger that is associated with love will lie somewhere over here.  
And so, and somewhere like anger, but that is also associated with hate lies somewhere over here. So the centroid, the point of centroid isn't actually where the anger is the most. It is the centre of different types of anger. The centroid now represents like you can be angry from all of these other emotions as well, but the combination of all of these.  
other emotions and the centroid represents actually the centre of types of anger and the max, the most angry emotion.  
will still be the belt around it.  
But...  
Let's go back to the centroid and how we calculated the centroid.  
We get every output from the LLM that said this sentence here is about anger. Yes, so then we say, OK, so that's we have a coordinate point because it said that thing is about anger. What's the embedding?  
for that sentence. This is the embedding, right? Put it over here. Repeat the exercise. Now we have all the centroids for anger and then we have an average, but we know that some of that anger is going to be closer to love and some of that anger is going to be closer to hate.  
And then it's going to be at some, we have some distance from surprise, could be closer to surprise. So when we talk about the, when we talk about this numerically, what I mean is that right before softmax, the logits which are output for every class, yes.  
may or will contain the probabilities for the sentence belonging to every class and not just one single class. Because of the data set, we know, because of our data set, we only know that one, one set, we can only associate one sentence to one class. But originally, if we look at it contextually, then the sentence might contain context for other emotions as well. Right.  
So just when we look at this, we're actually preserving that. When we look at this scatter plot, we're preserving these distances we get from the logits because we're saying we're not going to get the maximum score only, we're going to get everything. How do we preserve it? We put it as a point in a 70-68 dimensional space.  
So whatever the score was, it's preserved because we have the coordinates and we can say, well, from the coordinates, we figure out that is that how it works. So when we look at a sentence and then the class it belongs to, then the original data set didn't preserve it.  
But in the embedding space, the contextual information for every class is preserved. Yeah.  
So.  
Now, there's one thing in the embedding space, it's 768.  
768 dimensions, we're going to get a distance between, say, love, between...  
Surprise and anger, and...  
Sorry.  
Love and anger, love and surprise.  
The logits from the pre.  
Now, this is the bit I'm not clear about. When we get an output from the LLM saying, oh, that thing, the prediction is that thing was love.  
That's going to be a word, right? The LLM said this is the word. So it gave us an embedding that turned into a token that turned into a word. So right now, right now, our experiment is set in a way that we have a data set, which will have a sentence and the label already assigned by humans. Yeah.  
And we're using a model to embed the sentence into an embedding. We're not, I know what you're trying to say, that then we use the same model to use those embeddings to give us an output. Yeah. And then compare with that. But right now that part is not done, which I will discuss with you, but I'm saving to do next. So there's a pipeline and then at the end we say, okay, so this is what the pipeline is at the end.  
where we get the results to say that thing is anger. What we have in the beginning then is our tokenizer creates the tokens, creates the embedding for that phrase, and that's the embedding we're using. Say this thing goes over there, and we have the label because it was labelled by humans, and that's where the embedding sits.  
And then the second part of the pipeline, when the thing was trained, or maybe if it wasn't, we're just using a ready-made.  
we get the output that's going to be one of those five things and that's something to be discussed. Yes. So what we're looking at is the embedding space for the ground truth, embedding space for the ground truth. Super. There's another, I'm just going to pause this because there's an interesting experiment to be done as well.  
So this is what it is.  
You heard of Yehoshua Benjio, maybe he's a guy out there and he and he's been in the game for a long time, like 30 years since what you call it, back propagation became a thing and people, we can train these networks with back propagation. He was there at that point. Okay. And then so kind of at the start when people say, well, actually we can train deep networks and then the game changer.  
Change completely right, and that's why we're here today, because this thing is happening. So, this guy was at Joshua Bengio, that was his brother, a guy called Sammy Bengio, but their brother, so they're working together with Sammy Bengio. He's a researcher at Google.  
OK, now what is Yehoshua Benjio always said?  
is that networks are overparameterized, he said. And the thing about overparameterization is when you give, the network has far more parameters than it needs to learn something. And if that's the case, it's always going to memorize, it's never going to generalize. That's the debate kind of thing. If you give something far more than it needs, it doesn't  
It doesn't need to generalize; it can memorise everything. So, when we were in class saying, "How many trainable parameters has a single neuron have three? Well, how many trainable parameters has this model by OpenAI have one trillion?" You figure, well, that's lots of parameters, and then the other one we're thinking about training.  
like in the regions of billions of parameters, so maybe 2 billion, 7 billion, so that's going to be the size and the trainable parameters. So what this guy did in 2017, Sammy Bengio and a group of researchers.  
was they said, well, if the network can learn, has far more parameters than it needs, and we're using like at the time, 2017, a network with 10s of millions of parameters was a big network, and we can have like 10s of millions of parameters, and we're trying to learn MNIST, which has 60,000 examples.  
We can tell that we have far more capacity on the network than we need, but people throw it anyway. So this is what they did. They got the, what can you name that data set, some famous data set that started big data. So this is 2012.  
There's a group at Stanford, they get good funding, and they start using something at Amazon called Mechanical Turk, which is crowdsourcing on tap. So you go to Amazon, I need 100 people to label a data set. And they spent one year labelling ImageNet, it's called. That's a data set. It's A million examples, so a million images.  
with 1000 classes and they got people working through Amazon Mechanical Turk. It must have cost like God knows how much to do this. And at the end of it, they had a data set of a million images taken from the data from the internet labelled by humans. That's a banana, that's a horse and so forth.  
1000 classes. This is 2012 and that was like, wow. Now we have the back propagation algorithm, we have GPUs, we have big data sets, we can train these deep networks. And then things like, well, things always improve because of an improvement before, but now all the ingredients are there, the compute.  
the algorithm, the data, and that now things are flying, right? And now, five years later, 2017, this Sammy Bengio plus his group, and he's a brother of the Oscar guy, and we're all in there talking about these data, these networks being over-parameterized. They have far, they're much bigger than they need to be. They have far more parameters.  
And he said, let's do this. These data, these neural networks are learning about these data sets and they come to super high accuracies like above 99% is customary, they're hitting 100%. Now let's do this. Let's.  
Randomise the labels.  
The label for the objects, so we're saying, let's say we only have two classes, horses and bananas, and we're gonna randomise some. We're gonna say this horse here is a banana, and this banana over here is a horse, but it was random, so some bananas were labelled as bananas, and some horses were labelled as horses.  
but some were labelled as the opposite. Opposite of them, yeah. They look at a horse and you look at another horse and they look like horses for us. But the label that you're telling the neural network to learn is that horse is going to be a banana and that horse over there is going to be a horse and vice versa because bananas can be different. She's trying to see if it's memorising or it's generalizing. Exactly.  
So what did they find? That it memorised everything. It could still hit above 99%. That's when you showed it a horse that was labelled as a banana, it would say that's a banana, it's not a horse.  
So the next thing that we could do here, but I think that's a separate paper, we could say, well, we use that experiment to see how this thing is happening in the embedding space. If we get a phrase, because this is not semantic, it's language, and we say, I hate you and say, this is actually love.  
Where does that fall in the embedding space? Is DLL more robust to say, well, you're telling me it's love, but is it? I'm going to put it over there. That kind of thing. You know, it's weird that it would have to be a separate paper and experiment. Yeah, but I think it would be an interesting one. And I think that would be a quick one as well, if we kind of, what's the word?  
Specify it well enough with a single problem. Yes, we'll have to resolve with OK, but that that's that that's the case, and when we're training models right right now, we're training models, yeah.  
But maybe it would work even without training to say, we have this, the phrase I imagine would fall in the same place of the embedding space, irrespective of the label. But if we train it with the wrong label, I think that's when interesting things would start happening. Lesson, but that, but.  
I don't know, the model might get, it does sound interesting. I would want to see what happens in that case. Like, because a lot of said is about bias, right? Oh, we have to remove bias.  
But bias is a personal thing, right? What's bias for one person is not bias for another person. And I think it would be an interesting experiment to say, well, if, anyway, that's just something to keep in mind for future reference. Anyway, it's all transcribed there. We can say.  
write that experiment down for a future thing, and hopefully it will do everything for us. Okay, so, oh, by the way, I looked at that paper of yours, and I thought that you've written knowing what the neural network doesn't know. For this paper, we can write knowing what only the neural network knew. Knowing what only the neural network knew.  
Yeah, maybe, anyways, that's a separate thing, but OK, I need to rewind, yeah, where we knowing only what it knew, right, 'cause it knows things that we don't, and then it's like, and embedding space, yeah, only neural network knows, but now we know, we know as well, yeah, that's a good one.  
OK, so we were discussing the how the centre is around the belt, and no, yes, the density, yes, the density is the maximum around the belt region, yeah, and we were discussing how these points, and so the next set of experiments that I had in mind.  
Because this, all of this, what we discussed so far is what we have found and what we have, how do we say, not assumed, concluded, right? What we have concluded from the visualisations that we saw. But the next set of experiments that I had in mind was one.  
that one that you discussed where we use another model to embed the original sentences and then compare with the ground truth to see if the pattern remains same. The metrics with different models, right? The range for the scales might change with different model for the embeddings, but we should see if the pattern, the assumption we're making, the hypothesis we're making.  
from those visualisations remain. All right. I'm just going to repeat that in my own words, and you're welcome to repeat again in a different way. Yes. So we're looking at repeating the experiment with more than one model to verify that the geometry remains the same, even though the scales might change a little bit. So we're expecting that we're going to see the same  
pattern of decay, the same belt pattern, the same pattern with the centroids, and their approximate location relative to all the other centroids. The hypothesis is who observe the same pattern overall.  
Okay.  
And so, and the other set of experiment that I had in mind was using actually fine-tuning a model on the training set of this particular data set, right, and then running the experiment on the validation and the test set, and how we meant to do it is that let's say there's a...  
In the model architecture, there's a transformer here, which gives out a fully connected output of.  
And size use it, and then we have another layer of fully connected layers, which is of 768 dimensions, yeah, and then final fully connected layer of we have 5 classes there, so five, which is fine.  
Now, once this model architecture is trained on the data set, on the training set, for validation set, when we're doing it, we take this out. This is the embeddings in the 768 dimension that we use to measure everything and plot.  
This is the logits layer right before softmax that we can use to judge, okay, how much does the model think it is feel? How much does the model think it is anger? Usually we might try, we have to make sure that we don't overfit so that.  
this logic somehow makes sense. So for sentences, we get value like 0. We might get like 0.9 for this, 0.09 for this, and then all of this are very minute values. Okay, so I'm just going to describe what I'm looking at here. So this is a set of experiments. We start with an architecture.  
for a transformer. The way the architecture looks like, it's a transformer. There's a fully connected layer at the end, and then there's a dimension 768. And then finally, in the final layer, we have 5 outputs, and those five outputs should give us  
the scores for the five sentiments. What we want to do is train the neural network with that architecture, and then as we make predictions for every example in the training data set and in the test data set, we want to make a record of the embeddings in the spinal collected layer.  
and we want to make a record of the logits in the final layer and have those as data so we can then perform our quantitative analysis on these results. Yes. Okay, so that's another experiment. This 5 fully connected layer will finally give us a softmax out.  
from which we'll know which class does the model think that the data point actually belongs to, which is fine. This number right here will tell us how much does the model, now we're talking about one model, and so we get to see how actually these embeddings belong with the, how the model is thinking.  
We're actually going into one model, because our discussion about how right now whatever we are doing is comparing embeddings with the ground truth, right? But with this way, we're actually seeing what the model thinks, right? The model is thinking, and then what the model is seeing, giving us the output. Yeah, so just, yeah, carry on for now comment after, yes.  
So this stuff, so the five layer logics right before softmax can tell, can be used to cheque whether our assumption that something that is 80, 90% anger and 20% fear lies somewhere in the region between centroids between anger and fear. So this will be able, we can calculate that point Euclidean distance from.  
all of the classes and then try to see, try to, so this logic can be treated as weights and then can we see if weights to cheque how far they are from those centroids of these classes. Okay, so.  
Does that make sense? When we talk about centroids, we're talking about a 768 dimensional space. Yes. When you talk about logits, we're talking about 5 scores. So can you see the difference?  
Centroid 768 dimensional space, logic fiber. Yes, I know what you mean, but let me tell you.  
So if you look at this thing here, this is the architecture we're looking at. I just wanted to make a note before we continue that the experiment is based on the tokenizer that's generating the embeddings. So this is before actually presenting anything to a network. We're just saying we want to encode these.  
phrases as tokens, and then look at the embeddings. That's what this set of experiments is based on so far. And then the next step would be to say, well, we want to look now at the output of the network and how the network would classify  
a network that was trained, that might not have been trained, that was fine-tuned, that was trained from scratch. Every possibility that we can kind of go through would be good. And we have a month to do that, a little bit less. I just wanted to add one thing about this final layer. So when you say the logits, you mentioned 5 layers.  
So I think you meant 5 scores. Five scores. Okay. Five, not 5 layers. A layers that will have 5 scores for every class. And why I think that that may be used as weights is because even though we're calculating...  
So we just discussed the weights thing. Yes, that's what I mean. So you're saying they should be the logits should be used or could be used as weights because there are five logits, yeah, and there are five centroids, yeah, that we need to calculate the Euclidean distance of a point from.  
So what I'm trying to, my hypothesis is that, let's say for one input, we have one single input after a trained model. Let's say it lies somewhere here in our dimensional space. And here are some centroids around it. Right.  
And our model, let's just say that it's closer to this particular one centroid, because this is the class that it belongs to. And so the logics might say that this, the score for this particular class is 0.9 or something. And it will have values for all of these classes as well. So what I'm trying to say is that can we find a relation?  
Between this logits output and the Euclidean distance from the centroids, because ideally the 0.9, if it says that it's 0.9 probability to be in one class, then it should be very close to it. OK, so there's I have a comment to make about that.  
Let's say we get logits that give us 5 scores and we say the score for anger is 0.9 and the sum of the remaining scores is 0.1 such or. Oh no, so we're missing out. Logits are not probabilities. Yeah, logits. That's what I'm being given. No, logits are absolute.  
Yes, absolute quantities. So now we have a logic here that's saying that the most likely score is anger. So if the most likely score is anger, the highest number is going to be anger, and then they're going to be another four numbers. So now we have a logic here.  
and we want to know how that point would fall into that 768 dimensional space. So the next thing is, well, if the highest score is for Ringer and the lowest score is for Joy, it follows that the nearest point this prediction would be  
is closer to anger and further from joy. We don't know what that distance is, but we know that that's how it should be. Intuitively, that's how it should be. And then there are another three scores that say, well, it might be closer or further from love and anger and surprise and all these others.  
So now the question you're asking is, can we tell based on the logits where that thing would fall in the n-dimensional space? And my guess is that that's a hard problem mathematically to solve. I imagine we could say there are a number of points that would satisfy  
Yes, yeah, we cannot find out that there's a unique point and it could be that it's very computationally expensive to find that unique point. That's what the logits, but then there's an embedded, there's those logits would have come from embeddings, so at some point.  
There were embeddings, there were tokens, and then there were the logics based on this architecture. I don't know, actually, because I never designed it to that point, but let's say it's possible. My guess is it is. It is, it is. I can do, I can write, I won't even need AI to write the code for this model. I mean, I don't know exactly.  
But yeah, let's say it is possible. I mean, I know that it's possible to turn a regressor network into a classifier and vice versa. And that's straightforward. And I've done that for the thesis. And so my assumption is it would be possible, but I've never tried doing that myself. So let's say it is possible we can do that.  
Before we get to the logics.  
we'd have embeddings. So what I, that's how I imagine that architecture would work like, yes, that we would have to find embeddings in this n-dimensional space and then that that eventually would turn into tokens and that eventually would turn into our logits and we'd have the predictions.  
What we could do is what we mentioned here is we record the embeddings. Yes, that's exactly what it is. So once we recorded the embeddings, we say, okay, we have one coordinate and that coordinate sits here in the dimensional space. And my guess is the hypothesis is once we have that coordinate, we would know at that point.  
point where it lies and say oh it's in that the streets one the next one is the other one it's very far from those three and then the logits ideally should reflect that and say oh yeah the highest score was up thing it's closest to and I think that in itself may be an interesting thing to add.  
that we observed that the logits, the scores follow the geometry of where that embedding fell in that 768 dimensional space. That's what, that is the next set of experiments. Now, this does extend our experiments on text.  
Yeah, because we looked at first, we look only purely at the embeddings without even running the thing through the neural network. We're only looking at the tokenizer. Super. Yes, so, but that's, oh, that's okay. No.  
Now what the next set of experiments. So we have some hypotheses from what we've seen so far. Now, if we're able to train this architecture on the data, right now it's also pre-trained. So then all of these scatter plots are very noisy once the model is trained.  
I feel like that these some cluster for now that that is also something that I'm looking forward to seeing what happens to these densities once it is a fine-tuned model and not a generalised model. Do we start seeing them a clustering like the clustering comes closer, tighter?  
That will be interesting, especially with models that, for instance, there are some quen models that are the ones I've pre-trained. We're fine-tuned. Yes, fine-tuning. So the fine-tuned, these quen models, they basically, they give you all the tools. Like there's some deepseek models that they don't give you the tools. They say it's free to use, but we're not going to give you the tools to fine-tune this.  
Or which doesn't help, but there's some quen models that do exactly that, and I think that would be an interesting thing. I've done this for the self-driving experiment. OK, and that's the experiment there.  
No.  
You show the model of growth, and then you say, and this is a vision language model. When I say, should I be driving, should I be steering straight left or right? And then the vision language model said, though, you should be steering straight, you should be steering left, you should be steering right. And then  
They don't do very well. And then I fine-tuned it with like 10s of thousands of examples and it still didn't do very well. OK, but I left it as future work to work on because this was like I have to finish and send it and that's as far as I could get. But then even then I start having noise and say, can you still tell me which way I need to steer? Because that's the idea of the.  
Yes, well, that's the fine-tuning I did with this friend model, which was a language model, but I think our fine our fine-tuning experiment, this one, this would have to be a custom architecture, so we can do it from scratch with a custom architecture, no problem.  
So, do a lot, no, no, so this this transformer that we use here, yeah, yeah, that that will be pre-trained, uh, yes.  
We can take like something, the one we're already using, the one we are using to embed the sentence into 768 dimensions. Yeah. We, that's a transformer model. It's A pre-trained transformer model. We could just add these layers after it. We don't even have to, and then train it. So we're using a tokenizer, right, to get the embeddings. So at that point we only have a tokenizer.  
We haven't even looked at the model yet to say, now run this thing through the model. Yes. That's where we are, right? Yes. And when we, yeah, carry on. So what I'm talking about, so, so the transfer, that this is what we've done so far. This is something that we've done. We've used a pre-trained transformer tokenizer to embed sentences into 768 dimensions.  
Okay, so as far as I understand, tokenizers, whoever is training, so tokenizers are like a standalone thing. It's like a pump that can pump up any tire. So the actual bicycle doesn't matter. Let's say the bicycle is a transformer. That's how I understand this thing.  
Yes, but right now my architecture is using tokenizer and a transformer model right above it to try to embed the information. So the embeddings is, so you're saying the embeddings are generated by the transformer cell? Yes, there's a token, so in our experiment there's a tokenizer right here.  
Yeah, that gives out that that tokenize the sentence, and that goes into transformer. That's one which is a 768, and this is this is what we've done so far, right? So we get the embeddings from the transformer, so the tokens go into the transformer, it creates the embedding that we're taking. What's the model we're using here?  
It's right, it's the hundreds of millions model you mentioned. Yes, it's 500, 600 million per metre model. OK, so...  
So, we've done so far, and now when we're training it, you know the concept of freezing parameters, right? Yeah, yes. So, I'm going to unfreeze everything at the token, not the tokenizer. We'll use a good, I'll come up with the, I think the word the tokenizer that we are using right now is suggested from the previous.  
literature review experiment so we can trust it. But if you want, we can use, I don't think we should change the tokenizer at this point. We shouldn't touch the tokenizer. Okay, this tokenizer remains same. We should use the same tokenizer for the entire set of experiments. Okay, so the tokenizer remains this one.  
But everything before after that trains, everything is unfreeze. Wait, wait, wait. In fact, I want to take that back. For training this network specifically, so let's say we're using this network that has, I'm guessing, 600 million parameters. We use the same tokenizer for everything. But for the other experiments, we can change the tokenizer. We're looking at seeing.  
That the geometry is preserved independent of tokenizer, and yes, everything saying these are rules that apply, these are the this is the law of gravity of elements, and we can that applies to everything, language, single image, video, we observe the same patterns, yes, so...  
We can actually change the tokenizer for this set of experiments. Let's use the one that would work best with the model that we're trying to train. Right. Because I was thinking that we... One moment. So the model we're ready to train. I'm just going to get a bottle of crisps. Do you want another one? No, thank you. Miesha? Yes. Okay.  
Actually, I might be leaving from whenever 15 to 20 minutes, carry on tomorrow for, yes, 'cause, as I said, you know, this is continuous. We don't have to say, let's get to HPC working today. It's like, yes, it's continuous. So, for now, let's just plan it out and tomorrow we can, right?  
I'm gonna get my first Windows app. Yeah, go for it. If you get anything you want to know there, like the maps as well.  
And if you want some coke, get some in the fridge, no?  
Yes, so, in.  
Uh.  
In this architecture, we use the tokenizer that is, I think, I, I think, and just if I just ask the LLM to research and find me the best combination of these two, will be able to tell me, but I was planning on training a transformer bigger than 600 million parameters. Obviously, I want, I think we're able, we should.  
train somewhere around 1.6 or 2 billion parameter models because the models that are out there, LLMs, they're very big. And so at least to say that we know how this works, let's try to test something that's at least closer to that because 600 million parameters will be too less for us to.  
Hypothesis, and it's worth testing as well.  
But I mean, if we use a bigger model, we can at least say that again, now your conversion of memorising and not generalising comes to me. But okay, so that's. But the thing is verifying the geometry, right? You can use all sorts of time. Yeah, we can use all sorts of sizes, doesn't matter. So if we use a bigger model, train it.  
Then we take the 768 dimensions output out of it. We can, or let's just say the transformer doesn't output 768, it transports some output something bigger. We can use a fully connected layer to get it into a 768 dimension embeddings. And then from there logics, and then we only.  
Only this part will be trained.  
This is obviously frozen.  
And then if we do it on one training set, and we also would want to ensure that we're using training set and test set differently, because if we try to predict on the training set, the logics will always be like 00100, because that is something that has already seen.  
Possibly. I mean, let's just, but it's always best to see the performance of a model on the test set. We don't want it to remember from the training set the logic 0010. We want it to, hey, this is something new. What do you think it is? We don't want it to ask something that it has already seen.  
And because the training, the training data that we will use will have the logic says 00100, not logic, but train. This is the ground truth for that sentence. So the model, if it remembers it, then maybe the logics also turn out to 00100. I don't want that. I want to see how you do on a new.  
Sentence that you have not seen, where we can, where we can at least say that, yes, the model might be able to give a 0.1, 0.2, OK, something like that, but this is a very important topic that is evaluation, and it's a very important topic in the LLM space. Ananthropic, it says that in their interviews.  
They ask a lot about evaluation. How are you evaluating this? I'm not 100% because I haven't done evaluation with LLMs myself.  
how this is working with LLMs, but that's something to consider, like the evaluation, what's been done already, and we can get some deep research done.  
Well, it's going to be in a transcript, like, but yes, by all means.  
We want to find out how these models are evaluated. And for instance, in that transformer architecture I used, I, but when we did the transformers lecture with Artur the other day, we spoke about this paper called Attention is all you need. I read it.  
And then we read the abstract, and in the abstract it said...  
The evaluation metrics we use here is called BLEU, which is bilingual something. So that was the evaluation metrics they use. They have a model, they have a set, and the experiment is based on a label data set that said.  
The correct answer for this one is this thing in German. And then based on the answers the LLM gave, they could check, well, this answer is correct, this one is incorrect. And we came up with a score at so much using the BLEU evaluation metric. So there is this. So if you write the paper and you said, I use the BLEU evaluation metric, everyone will know what it is.  
So now we come to this important point of saying, well, what's our evaluation metric? And then you say, well, we used an evaluation metric like this. The best thing would be to say we used an evaluation metric that's used already in these cases, but it could be, but this whole idea here is novel, say, well, we actually, we modified  
Because we're studying specifically geometry, this is what we wanted to see. And what we want to see is how the logic relates to the embedding that we're plotting in the n-dimensional space. And that then justifies the choice. And if people say, well, we use actually use something that didn't exist, we say, well, but we use it for a reason.  
And that's the reason we use it for. And also, and also what, and also this makes sense because the logics is actually what the model thinks. It's like, this is my output. The softmax that we do is a mathematical calculation and not, it does not come from the model. Does it make sense that there's no neurons that have given us the softmax layer output? Softmax is a simple mathematical calculation.  
It's a normalisation technique, so we have to compare what embed how those embeddings are related with the logics and not the softmax output, because, as of now, we were comparing embedding space with the final output softmax output, but if we talk about...  
How now, because we want to see what the model is thinking and how the model is thinking, we have to trace it like we have to trace everything between the model layers.  
And we have to look at the last layer of the model itself and not like this, so something before softmax layer. We have to look at that and then something between the model to see how those things are related instead of between the model and something that we get after a mathematical calculation over the output.  
Put off the model, say that again.  
We want to compare how the embeddings, which is the middle region of the model, is related. Which is what? Somewhere embeddings is somewhere in the mid region of the model. You know, that's hypothesis.  
The embeddings is somewhere in the mid-region of the model.  
OK, somewhere between the somewhere inside the model, it must be because it's the embeddings, right? So, not I won't say you're correct, I should not say middle region. I'm saying that embeddings which is inside the model.  
And how it relates to logics, which is the final layer of the model.  
1.  
Because logits is the final layer of the model that comes from neurons multiplication and comes from neurons. The softmax output that we get from the logits is an output of a mathematical calculation over the neuron calculation. Over weights, right?  
Over, yes, overweight, so yeah.  
So, so that's an important part for this evaluation for this set of experiments that we're doing, that we that we will keep on looking at the logits more than the softmax output. Well, the softmax output, I think.  
It's there if we need it. We have a record of the logics, so if they're ever needed, the softmax can be generated. Yes, and it's a referee set. So why don't you use the softmax, and we can quickly say, oh, we added it to the appendix. Now you can read it. Yes, so the softmax output is only important when we want to look at.  
Okay, so which class does the model think it belongs to finally? But that we should get from the logic as well. Yeah, that we should get from the logic. The highest score is. So it doesn't kind of end things. By the way, I just wanted to mention that when I was doing this stuff, this is going back two years because this has been in discussion for two years at least, maybe a little bit more or less.  
One of the lecturers here, Tillman, you might know him from data. So I had some discussions and some arguments with him as well, which is not a bad thing. There's arguments. I think it's a good thing to argue as well. And then I told them it could be down to cultural differences and we argued A lot. And one of the things he said was,  
you know, you use the softmax, which is true, that's the name of the paper, exploration of the softmax space. And I looked at all types of metrics. Once we have a softmax, what else can we use? And I said, well, we can use other types of distances, Bhattacharya. And then he said, but it would be interesting to look at  
The Lodges.  
Because the logits, when you're normalizing, you're keeping, but you're losing a little bit. Yes, yes. And then I said, yeah, that would be an interesting thing, but I never did it. So this is a case where the logits kind of. When you've done, when we've done it, maybe you can go back to Tillman and maybe.  
Well, maybe go back to that thing at one point, but it's like, it's, but I think it's the same thing. It's looking at this end-dimensional space and saying, well, we're not going to look at softmax, we're going to look at logics. We want to know what those absolute values are. Yes. Because the absolute values should say, okay, there is something we need to know about the absolute values of anger.  
and love and so forth. We have to know. So the only difference, I think, what we're losing when we do the softmax is that now where the softmax output will compare all of the emotions within each other, but the logics, the logics only has.  
How sad do you think it is? How angry do you think it is? But we do the, when we do the softmax, we're trying to compare, okay, now compare all of those values and then make it to one. And so maybe, maybe we don't want to look at all of the other emotions when we're trying to look at just one emotion. Does that make sense?  
Well, the logits are of course for every emotion, right? So we get that information, not in the sense that the softmax gives us, because with the softmax, we could say, oh, we, okay, this is 90%, and then the other two are 10%. The logits might say, oh, this one is 60 and the other are three.  
and we'd have to work it out ourselves. Yes, exactly. That's what I'm trying to say. But because it's an absolute score, we could say, well, we can see that with love, that love is scoring at 80 and then anger is scoring at 30. But when we look at surprise, we can see that when surprise is scoring at...  
40 anger could be scoring higher. Now that thing we lose with, I think we would lose with softmax to be discussed. But one thing we definitely would lose is the absolute value, to say that the highest value we're going to get for love.  
Is X, and the highest value we're gonna get for anger is so forth, and we would be able to say then, and this also, this also, this will also hold true because when we looked at these embeddings, the raw ones and the normalised ones, we saw how raw performed better.  
In these things, because, because so when you say things, that's something that this transcription, it won't be able, it might not be able to work out set of experiments. Yes, for our set of experiments, we saw that using raw embeddings from the model work was better as compared to the normalised embeddings.  
because the raw embeddings had the magnitude, which is kind of important because we're trying to understand the context of the text or when emotion. So if something is very angry, we don't want to lose the very part of the angry. So it also works the similar way in the logits and the software.  
So, if the logics are like 80, 50, 10, and one, we want to make sure that this 80, one, 10, 50, this, these, they can be treated as magnitudes right here, the the soft max, you mean the logits, the logits, the logic for magnitudes, yes, yes, the logits will have the magnitude and direction.  
better than the software output. So I think the great advantage of the logics is that we can compare to some inter-class comparisons to say that that's the strongest feeling is anger to be able to make that kind of statement. Yes, anger is.  
Stronger than love, or love is stronger than anger, or anger and love are equally strong. I think that would even be a good reading for the paper, and then we say conclusions like anger is stronger than love. Ohh, yes, yes.  
Where anger is has a higher magnitude or a higher absolute value, but that would be for one sentence, yes, for one sentence.  
But perhaps, but then you have average magnitude as well, that on average, anger is a much stronger emotion. And make that kind of statement if we have absolute values, which I guess you won't be able to say, you won't be able to say with softmax, because it would always be a percentage.  
So essentially what we're trying to study, it's important that at all stages we look at values that have both magnitude and direction. We don't want to lose magnitude whenever we validate. That will be a good way to generalise this, I guess. We don't want to use magnitude when?  
No, there could be, we, whenever for this set of experiments, yeah, in every value that we're comparing, we want to ensure that the value captures magnitude and direction both, and not just direction.  
So, the...  
Embedding should give us magnitude and direction because it's a vector. Yes, yes, that's what I'm trying to say that in every value that we're looking at, it's better to keep the magnitude in.  
OK, whether it's a logic, embeddings, embeddings, we want to keep the magnitude, yeah, we keep we want to keep the magnitude of these vectors along with the directions.  
I think, I think, yeah, and then this is the next set of experiments we can work on. Build the network, train the model, train the model, pre-train or fine-tune, train from scratch. So training from scratch, that's an interesting one because you need a massive data set to train from scratch.  
we'd have to find or define ourselves a very small, so let's say you have 10s of thousands of examples, I'm guessing, could be thousands, could be 10s of thousands, but if you have 10s of thousands of examples, my guess is that to get a meaningful set of results here, we would need an LLM that would be.  
in the region of 10s of millions of parameters, which is a tiny thing. I think GPT1 had 10s of millions of parameters. So something on that order of magnitude, as far as training from scratch, which I think would be super interesting. So we trained the model from scratch, this is the set of results, we fine-tuned one, this set of results and so forth.  
So that's, yeah, it sounds like it's going to be. That's why I said that we will need HPC soon. Yes.  
I think we can work on that tomorrow, I see. Yeah, we I think we should, and now we just leave experiments running until the end of the month, right? Yeah, yeah, and then we have to, yes, because I was just thinking that it's 11th April today, and we wanna can we wanna be at least like text is good, and I don't mind doing this experiment on text because...  
Now that we've done this, we have a whole set of experiments ready that we can just repeat on the images directly. The segmentation is just a pre-processing, so it should not take a long time. And then we can just repeat the whole process on the images and then you don't have to. And I think with the images.  
even without segmentation, could be kind of an interesting experiment to say this is the first experiment. Yes, the date. Yes. No segmentation, segmentation and so forth. Okay, sounds good. So I think that if we have everything, everything, all the experiments by the end of this month, we're in a great position because then it's just stitch everything together, send the paper off.  
Yes. But the deadline is 4th of May. Yeah. So it has to be the end of this month. That gives us three days to actually write that. Well, I'll start with the template. I'll share you in the overleaf, and then you'll see what it is and where the overleaf is, and now it's progressing. But the overleaf is basically these two transcripts we have.  
that starts to build it. And then based on the transcripts, we're looking at the literature survey, which we have already, and the rest is like just, so this paper is factually written. The bit that's missing is the experiments, the results, and the discussion. That context that we get from here. Yeah.  
So, and then conclusion and future work, we see where this is going already. It's going to videos. The conclusion is the geometry is similar. Whatever you do, you observe these. And these are the kind of rules, the law of gravity in this dimensional space. And I think the paper from that perspective, that's the bit that's missing.  
It could even be, I don't want to do it, just a conceptual paper. We have, we conjecture that this could happen. It's A conjecture. And these papers like this, but that's not what we want. We won't, we won't put, we won't go on the stage and say,  
I think this, no, I know it, yes, because here are the numbers that prove what I'm trying to say, so I think that'll be better if we do that. So, and so just to recap, consider the papers written based on these two transcripts.  
And then the you need to answer a call or something. Go ahead. No, no, it's just I he's asking if I've left or not. OK, so yeah, go for it. I mean, no problem. I'll make a cup of tea. Ohh, you have to leave like as soon, right? It's at half past. No, I have to leave at 8:45. oh my god, let's get going, man. Yeah, yeah.  
So yeah, it's good, good progress. I'm gonna stop the meeting now. Yep, so we we have to transfer things.  
Of transcription allow the minutes and the transcriptions to the GitHub, like I did, so we have a record on GitHub, and then we can meet again to start running the experiments on the HPC.  
Super. And you want to do that? Do you want to do it tomorrow? The HPC. Yeah, we can start tomorrow. Alternatively, anytime during the week, because I'm not teaching anymore. So the weeks are like blank. I can work like on this full time until we get this paper written.  
Okay, so it's up to you. I mean, oh, I am coming in on Monday evening because weekdays I will work. No problem, yeah, go for it. Oh, so 12 to 6 is the time that I work on weekdays, right? But after six, if available, because...  
The time that we met today would be optimum for me and then I can come in on weekdays as well. Okay, so my calendar for the week is I'm available except Monday and Wednesday. So I can do Tuesday evenings, Thursdays and Fridays at the moment. So, oh, but Tuesday I have a commitment.

Sikar, Daniel** stopped transcription
