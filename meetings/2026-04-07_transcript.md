**Transcript**

7 April 2026, 2:55pm

Sikar, Daniel** started transcription

PG-Verma, Pritish Ranjan** 1:51  
OK.  
So you downloaded the data sets. Yes, let me. So what I did was yesterday I thought of starting with the text part a bit. Remember last discussions? Should I come closer? Yeah.  
Sorry.  
So, yeah, you're saying you remember? Yes. Last time we were discussing about the text data set selection. Yes, we were going through prompts from Codex and Gemini. So I had, I got that done and the results. Let me try to share my screen.  
Yes.  
It's over here like you look at the screen. I'll fine, I'll just turn the air around so I can look at the screen. OK, that makes sense.  
I guess we're both going through lunch right before this, yes.  
Let me open it up.  
Right, so this SST two data set, this was there in Gemini and Openai, both of them. Yes, so I downloaded that and this is how the data set looks. Let me just open the test one so that we can.  
Classic, like classic, yes, yes, classic sentiment analysis data. And it was a binary one, so there would be.  
A positive and negative sentiment with every so. So these are 0 means negative and one means positive. Yeah, and these sentences. I mean, I went through some of them.  
They some of them were a little vague because I think they come from a common collection of text and this is divided and so some of them will look a little vague but mostly it just like for example this in world cinema.  
As we go along, yes. So the data set isn't clearly widespread apart in the sense that the two classes may not be very far from each other because the data set has some vague data points where you know some something like in world cinema will not exactly be positive or negative.  
But it is termed as positive, which makes sense. But then again, it's not clearly some a positive sentence. Interesting. But does that follow the space? Yes, exactly. That's what I'm considering. That would be somewhere closer to center or somewhere far away from where we want it to be, but might not be exactly happy. So we can.  
I I've done the first level experiment, but we can try filtering out to find only the exactly and you know appropriate or less sentences. That would definitely mean one thing. That would be interesting as well to see the cluster is here. The happy cluster is here.  
The SAP clusters, yeah. And which vape one if it falls somewhere in the middle. Perfect. Yes, that's that's that's the. So that's good because I've done the and then and then, yeah, so that was about the data set.  
And then so in the vid IQ this one I've created experiments folder inside which I've defined text first this for I'm I've done 2 experiments now binary and multi classes one in binary the SST two data.  
I said that's the one I chose. I'll go to.  
Report.  
Yeah, so for. So in the plan, in the plan, let me open it and preview and then do this.  
So I asked it to 1st create a plan for me. I I gave it the reference for the SST two data set and I told it my whole plan. You ask it to create a form to create a plan just to create a plan just to create a plan instead of prompt. I thought that.  
For the whole scope of experiment from for our whole experiment, create a plan for me.  
Like where I want I want to create the embeddings, how to store the embeddings and then final validation. Ask to create a complete plan for me and then tell me where all other variants where as I can try different things. So it told me the model selection is where you can.  
Try out different things. Then the second part was embeddings. How we pull the embeddings from the from the model output that can be something that has multiple ways to try. So I told so basically I told her to create the plan and spread out what all experiments can be done like different things I can try with it.  
So the first thing was model selection. Where is the date? I gave it the whole context of the data set, my experiment, what I plan to do, whatever we have done so far. And and by the way, I've also given the reference to the 80 papers. That's why I told that in the papers because I asked it to go through every paper first.  
List out which one's relevant for this experiment and then and I have. Then I asked it to go through the relevant papers and then a frame of the plan for me so that it knows the things that has been tried in other papers. We know that we are. We won't make those mistakes that that has already been discussed in those papers.  
Theoretically, well, the mistake is one way of calling it in a paper. OK, that these experiments have been tried and we've noticed through development experiments. So that's what we need to keep in mind. OK, I'll keep it in mind. It's OK. The experiments tried would not be.  
So in that I gave the context of the pay research literature survey that we did and then the first was model selection. We went through two models, these two models and then out of which.  
Like the base, this is the one that was. Oh, sorry.  
This is the one that we ended up choosing with the output. At least I did that time.  
And then from the model output, there are three ways to pull the pure omelet to go. Now just break eggs and make omelet. Remember the election to make an omelet. You need to break eggs. And you're like, yeah, I chose this model here. Let's go with that one to start with.  
Yes, see how far it gets. Yes, exactly. Yes. And then I was doing this call at like 3:00 AM or something when I was emailing you anyways. So from the model output we get three, we we can essentially take out the embeddings in three different ways, draw one which will have the.  
Magnitude and the direction of the vectors and L2 normalization is the one where we normalize the magnitude of those embeddings, not the directions. This is good because that means you can compare phrases different. Yes. So that means that if something is too sad.  
But there is also like in the raw one, it will be too sad. The magnitude will be there. But let's say if they're too a little bit sad, too sad, then there's there's another thing. Sorry, come in. No, no, no, go ahead, go ahead. The sentence is that long.  
It creates an embedding about that one. That's kind of the intuition. If a sentence is that long, it creates embeddings about that one. Mm-hmm. Normalizing. It means that both would be the same size. So it's a good thing to normalize. OK, yes. So although you'd be surprised because the results came out that the log one performed better.  
It did. Wow. Yes, you can see that. You can see the interpretations from the output results. I asked it to note it down and report everything. I asked AI to not like store the results. What are my interpretations? What should I decide on the basis of that?  
Yes, there's more there. This is just for binary. There's also the same thing done for multi classes one and there's some results that will surprise you there as well. Anyways, in this then we plot the results.  
Is how the clusters look.  
This is the history. I'm sorry, the clusters you can see because it's sad and happy, although it's all. Also I've done dimensional reductionality to plot it because otherwise it wouldn't have been possible to plot it, but I'm hoping that this shows that the cluster does.  
There's a cluster there.  
So they do work with the meaning.  
You know what? The Hits in itself would be a novelty. I don't remember seeing this in newspapers.  
Like plotting them out with dimensional could be a normal thing, but the plot reduces the dimension from 768 which are my embedding size to like it reduces using PCA.  
So there's some information that might be lost. There's another thing we could do. We could keep the 728 dimensions and do and do.  
That's the word line clustering. They find the two centroids on those two clusters. So the clustering done is on 768 dimensions. It's just plotting that has been done. I ensured that all the clusters and the metrics that we get should be done in the 768 dimensional space.  
And not in the two-dimensional space. There's a lot of information that will be lost from 768 to two dimensions. For instance, one thing and then what we can do here is we can say what is the centroid of this thing here? It's gonna be a point around here.  
Then we can start talking about the laundry.  
So in the normalized space, the maximum distance between any two of these points is sqrt 2. Yes. And then we can say, well, if we get the center of this one, what is the furthest positive from the center? It's so much and then what is the? And then we do the same thing for the negative.  
How tightly are they clustered around this thing? You can start asking those questions, but that but but that we split with that in 768 dimensions. Yes, yes, not too, because not too, no problem. Because I can't ask you these points being somewhere like this. We don't know.  
I've been doing good kind of like it's about 100 donations. We can't split. Yeah, if we can, we can do very interesting.  
And then we move on to. So this was the result. The results showed that yes, where is that?  
Exactly. The raw pooling was the best way to do it. So that means that for text in binary, we found out that normalization didn't really help us. It wasn't correct. Yes, I mean the results were very close, very close. Even L2 normalization was not very far away.  
But raw was a little better.  
OK.  
It's gonna be different. I never done the distance withdraw, but it's it's doable 'cause it's distance, I think, yeah.  
Wow.  
OK. Thank you. And then we move on to multi class.  
This experiment kind of remains same in the sense that oh also for model selection I chose to decide on the go with only 10% of training data and all of the validation data. I created the embeddings and then I then I decided just from the embeddings without validation.  
Which ones? Which one would be better for us and how I did that level? We did 10 percent training data, 90 percent validation, 100 percent validation for model selection. This is not for embedding creation yet. I mean I did create embedding, but this is just to pick one which model is better.  
And for the whole experiment, we go with all of the training and all of the validation. That's really taking 100%. Have you seen that as as the agents of this exists in these papers before doing this?  
Select models. Uh, to be honest, just to select models. The the reason I'm saying this is because.  
The people who are going to read that New York paper, they're referees, and they're the type that ask these kinds of questions. Why did you do it that way? And then as long as it can be justified though, because these authors did that way as well, or because I tried it like this, like this and like that, and I saw that it didn't make much difference. So I went.  
100%. The thing is that our referee might pick up on that. They might not, but they might. And you always have to be careful with everything you say, because if you say something and you you go to a paper like that, is there anything here that I can't back up? You need to actually.  
I'm not saying fix it now. No, no, I'm just making a note that it's something that we have to always keep in mind when you submit papers to conferences. And it's the same thing with the RSC different page plan, although it's going to become it. But one day if it's going to be the same thing.  
You don't wanna say why would you do it this way? And if you say, oh, because I thought it was a good idea, that's not, that's not a good answer. It has to have some kind of OK. The reason I chose to do that was because I wanted to save some computations. That's fair enough if you I wanted to save from.  
And then you describe what these things are. I was looking at simplifying the process because if I selected the model through a different thing, it would be very computationally intensive. So it's back to the random selection. Yes, so I did that because.  
Model selection is an easier task out of the two. We just want to check which embeddings are more contextually, you know, right? Correct. So for that we don't have to go through all of the data, we just have to pick at random 10 per like a certain a fraction of.  
Of the data so that we save some computations and we can speed up the process of model selection because there's multiple experiments that we do in throughout this and we're not just selecting model, we're doing further on. We're also selecting which way to pull the embeddings from and then.  
The correct validation metrics is. So the main task there is to study embeddings, not to yes the models. The main task is to study embedding. So so it justifies like not being too careful selecting the model. Yes, I mean I would rather spend more time selecting on the type of.  
Trick to validate those embeddings than to spend time on models because all models nowadays perform good at keeping semantic contextuality. And the main thing is to validate the metrics, not to validate the the models. OK, that justifies the choice.  
OK and so then the multi class one.  
I skipped the model selection embedding run for this one because we already had a model. OK, fix starting point please.  
Yeah, I've written it down as well. Everything by model selection is being skipped. Every all of the all of my experiments are reported.  
OK, so essentially the contract to use codex, yes, with your with your plus, yes, yes, and they covered all your queries, all your prompts, yes, but I but I'm done with my tokens now, so.  
Oh, you're done with your tokens. I used up all of them last night. That's why I slept until next month or for this week. OK, so don't reset. I use Gemini, but OK, it's not good, but it works. Yeah. Wow, you burnt out all your toks. That's why I slept.  
So yes, so the plan for this one remains same.  
So.  
I was just gonna say, hey, come back.  
OK.  
Uh.  
So the plan remained kind of same first. First there was data set preparation and then embedding generation and then the multi class validation. Most of the metrics remain the same from the one we did in the binary one, but the only difference came out in the results where now.  
Yeah, I'll just show you because it's not as good as previous one.  
Now this happened. I explored it kind of why it happened. Then we found out that clusters like joy and love, they come together. Yeah, they overlap. Anger, sadness and fear, they overlap. But that's interesting because with the geometry.  
Get all the surprise, fear, the love, joy, sadness and cluster them and say we just average the the embeddings and say that's where the centroid is and study the overlap and then we can talk about volumes.  
And say the volume between fear and surprise is this much, and the volume between fear and anger is this much. So my hypothesis is that there will be a bigger volume between those two. So let's see.  
You are fear, and I'm surprised. And then over there is, say, meal. You meal with sadness. So let's say fear. It's more, I think, closer to sometimes because in a state of shock than in a state of sadness.  
So the volume between these two would be bigger. That's the hypothesis. And then if if that turns out to be true, which I believe it would, then we can write that in the. It's the smallest self price. It's everything. What's that? See that? I mean the self price data points. How interesting.  
It's everywhere. It's in love. It's in. So that's kind of the theme, right? Let's see. But you get the idea. And then you say, well, how to spread out your these points with?  
Compared to centric and then we can get to these geometric studies. But yeah, let's say surprise is the one that's most spread out. So if we look at the surprise cloud, we'll say, OK, it's very kind of sparse because it's spread and it could be that.  
The surprise here and then here and then and then you can see a cluster kind of. So there's some separation with this green surprise. It's heavy. Interesting. Fear is fear clustered somewhere and then another thing we could do for the paper because we could have.  
Have the paint of these small ones and compare them with this. You know we have two plots to say we consider these two and then we have like the ones we want to show something to say what surprises mixed with everything. There's a plot for surprise in New York.  
I think it's even to me, so that will be a little bit separate. I'll I'll write it down.  
Stupid.  
That's repository is looking so mean. It's like if if this is going to view it, this might work, but it's.  
So I'm giving context of my project description there. So I what I'm gonna be doing in these things is that I create context for the whole project and then I create separate context files for separate project. So in that sense.  
When I'm writing a prompt, I don't have to write anything, but I'll just say do this context in this file, do this context in this file. I just I just write exactly the sentence that I want it to do and then I get past the file that will have context of what it needs to do and all that nice.  
That that I'd like to take one of that important context before you.  
Context is what you need. Context prompting. Yes, but no, it's it's a it's a joke on attention is only yes, yes, but that could be the technique, right? Context prompting. All you have to do is say this is the prompt and the context is over there. Yes, yes.  
Uh, so yes, what we need to do is.  
Plot overlaps like 22 classes that are significant and maybe have a block of four or block of dates.  
I'm feeling like a PhD supervisor now. Mind you, there's not many supervisors like that. There's one that they that's all they discussed. They sit down and say these are the experiments we can speak. All the other ones are up there and we want just to leave the meeting as soon as possible.  
That's I was. I don't know. I think about. I've been thinking about the PhD, but I think.  
Trust me, it would be so easy for you to do a PhD. Tell me, how long were you doing this for? Five hours. So that's it's like 6 months, you know, for PhD. Six months. Your new PhD is all easy, right? But it's like it's it's just that I don't pay in my own.  
Like I went right now. The thing is that I'll I'll be frank with you about my personal situation here. Yeah, you told me you have to find a visa, right? To sponsor, not not just that. And I mean, I could get a visa if I apply for PhD. I could get student visa.  
But The thing is that my mother, she's back home and I don't have a dad. I lost him when I was seven years old, right? So my mother is still working minimum wage back in home country. So it gets like she's a little dependent on me. I I have her and my younger brother.  
So I've been handling financial things for a while now and I think if I do a PhD then I won't be able to provide the kind of support that they need. You can't get into a PhD and go look for a job straight away. This PhD can do one day a week easy.  
Trust me, man. Can I do it with this job? Yes, I I mentioned this to you. We forgot. But no, was it for you or was it someone else? It was someone else. I get a PhD with a job and the way the only time someone said anything was when the taxman.  
Contacted me to say you haven't paid another taxes. They said you owe us this extra.  
And I said no problem, here's the money. And that was that. That was the only time they told me anything.  
OK, then then that's I think then if I can secure a job with it, then you'll find 'cause then you'll write to you at the end of the year and say, oh, you haven't paid enough taxes and then you just pay them, but you haven't only end up lending anyway, so it won't be a problem.  
That's anyway, that's my advice, you know, because you're going to do well with your PhD, man. And another thing is this project of yours of the video, that's a PhD, right? That's a PhD. That's a PhD.  
No one.  
Get on to, I think. Anyway, I think you should think of our. Think about the CVT. They have PhDs upstairs.  
I think Merala is in with that. She knows and so I think would I be able like there's, I don't know, would I be able to do freelance jobs as well with it for products, consulting jobs or something like that? I'm not sure if it seems we do anything like as long as we turn up here.  
Then talk to your supervisor. If they ask you to turn up to Steven, then we always ask. No, if I'm if I'm doing PSD, I'd do it under you by the right. Great, but I don't. I might not be ready to supervise by the time I might be so when the time comes.  
But you still need funding to do the PhD. CDT has funding and they're a bit weird 'cause they want to meet every week. But you can talk to their people and say how often do I need to turn up here to do this PhD? But that's the salary, man.  
They're paying the salary every month. How much is that? I don't know. Maybe. I don't know. You have to talk to them.  
But then you can work as a GPA if you want each. Yeah, like I do.  
So it's a career as well on the site and loads of teachers here they are companies.  
They have company. They have company. OK, so you have to assign the security guide and I'm involved with.  
And he does consultancy work for companies. Yeah.  
You're not going to do anybody. Anyway, he's doing it really well. So you can just go for everyone, apply for the page. See if you can find the paid one. Worst case is saying responding for me. I'm sorry. Can you pause it? But we've got to begin.  
But I tell everyone.  
Try to give it a go 'cause the theme with you is that you're not, you're going to look at what you're doing already. This is PHC level stuff, you know, it'd be super gonna be easy for you trust me. But anyway, let's carry on this. Yes, right. So even a multi class thing and.  
This is the multi pass thing there was we have to calculate same points and plot overlaps in this one. The results were kind of kind of similar. Let me just go to results as well.  
The numbers weren't as good as the ones for the two classes, two for classes being close. I mean for two points and a class could be close and we're comparing the the accuracy of how good a class was in binary. Yes, between kind of the intuition, yes, the kind of kind of that's what we.  
Expected. Yes, the model and the pool, pooling selection remains the same. The best one remains the same, which is that model instead of. I'm just gonna check that I think mentioned.  
Neil and we were in HG 06.  
I think they said they you don't want them.  
From the meetings at AGO one and then that's what chat with AGO saying something like that, sure.  
You will be in the day. OK, I'll come tomorrow's translation. I'll talk to her.  
I'll be in tomorrow as well, so we can both talk.  
No, it's tomorrow's. It's tomorrow at 6:30 till five, starting 4:30 to five. All right, so.  
So you said, and then there was this thing, it was more spread out. It was what we were expecting. Yes, and you see Joy was a global and there's a facial, but most wrong with that. Surprisingly, it would be this. It's used to be different, right? Interesting.  
Which is also kind of. I thought that that makes sense because you can be surprised in any state, in any state, right? That's the same for the paper, right? Intuitively, fear can surprise you.  
Surprise is surprise is more of a reaction than an emotion, yes, so it can be. That's interesting. That's a very good qualitative analysis. Surprise is more reaction than emotion and hopefully don't be in the transcript. And then all we have to do is give the transcript to the AI and say finish.  
Polish the paper with the discussion we have. Oh, that's why you're transcripting everything. Yeah, of course. That's amazing. I never thought about it. It should be in the transcription in the meeting. There should be a transcription in the chat, so.  
It should be, oh, what's going on? Is it not transcribing?  
Oh, God almighty. All right, so we're not going to start again. Catastrophic. I'm sure I was reporting this more.  
We discussed so much.  
Maybe it's it'll show in your system.  
Mhm.  
Mhm.  
Oh, no, no. It's transcribing. So no.  
It is transfer and transfer.  
Right. We haven't. Oh, the same, the same. And do you know who's going to put in the paper? No, if you get some jobs, no problem. And the appendix. You have to be here. Yes, we will. Don't worry. You'll read your very sentence of the paper.  
Because that's why I do all this context work and I keep because when I ask the model 1st to do something, it'll do something so catastrophic that I have to keep on checking it. So I have this relationship with DI where I don't trust to discuss it. Oh yeah, it's all here.  
So good qualitative analysis, surprise in any state. Superb. Surprising model of reaction and an emotion. Brilliant. So we have it all. So everything we discussed is here, right? So yes, it's. I didn't even think of that. That's so.  
That's the context is all you need. Context is all you need.  
So, uh, OK, so oh, let me go back to maximizing upstream.  
Oh, then you have some. So you were saying surprise with everything. That's the weakest class. Mm-hmm. Is this something the AI wrote? The weakest class? I think the weakest class. Yes, that's the. I wasn't familiar with that concept, but I can see where it's coming from.  
The one that's more spread out, more confused, mixed with other others.  
'Cause there's a paper I wrote. Some classes are more likely to be misclassified than others. You know, with my first paper, and basically that's that that paper makes sense. It seems interesting because surprise is the most likely to be misclassified. Yes, because it's.  
Because it's a reaction and.  
Geometric interpretations, dominant taxes, but no six way clean cluster. Interesting. The SMD folder outlined partially overlapping globs, which is expected when classes share effective meaning.  
What's effective meaning?  
For being determined, centroid heat maps confirm the pairwise distance listed in the summary. The overlaps are a real much of spiritualisation on fact psychometrics that computed in the native 768 space. Amazing 768 dimensions.  
Decision keep that model rolling and polling as the most past people to reference L2 with central dots variance with stress tests and take shifts. The next addition should be TIDF.  
Plus logistic re direction. Next place. I know what the idea is, but I don't understand why it suggested that. I don't know, but it's interesting that it's suggesting it in the third place. The idea is the idea is the score of how close.  
A topic is. So as far as I remember, the intention is if you have a dictionary and you have a load of massive amount of text and there's this one word that occurs one time in that whole text, that word is super relevant and that that TF IDF score would be very high. Now if that word occurs.  
100 or many times it wouldn't be as relevant 'cause it's not telling you a lot.  
So for instance, in this discussion we're having the word clustering, let's say it was used a little bit, not a lot. So the TFIDF score would be high. That's how I remember the intuition of TFIDF. I'll go through that by accessing that. And then the other thing that is incident was Max sled 128 because.  
Doesn't truncate some of the text. I checked for the whole data. Yeah, in the binary one we only truncate, we only remove 0.2 percentage, so that's OK. We're not doing a lot. In the multi class one we remove around 6% of the complete data. So I was wondering if that.  
But is this truncation not the length of the embedding vector? No, it's from the text itself. I was thinking from the text itself might be a bit dangerous because you're saying, oh, we're going to.  
Uh, we move text that could have me. Yes, it is dangerous, but I checked for the whole data set. We don't remove. We don't remove almost anything for the binary one, right? So I can do the experiment by increasing the size and length of size that we accept in the model maybe.  
But for the binary one, it would not help. I'm not going to say we should do this, but I'm going to leave it as a future experiment here in the transcript that potentially what we could do is we could get an LLM to say you're going to have to expand or contract everything here so we can say that same word, that same phrase in seven words.  
And everything is gonna be 10 words long.  
And look at that. So can you see where this is going? And then we're just we're leaving it to another level to decide what length it should be to be the same meaning, the same like sentiments. And then there's not a lot in in the hypothesis is.  
That the embeddings are going to be more meaningful because the length of the phrase is the same.  
That's the hypothesis. I'm gonna even say we test it here with some idea. I might even try this out. Yes, but you can you can you can go on and do that. You can go on and push your experiments and right export and you know the comments and every yes comments or reports and.  
So that's what I've been doing. I I've I've I follow what you taught me. So there's there's prompts, there's reports and there's work diary for everything. Super so and although because I use context and not prompts, so the prompt folder is empty, but in the report you will find.  
All the experiments that I've done and how I've done it right so.  
Next steps, I guess that after this we can start looking at videos, images, yeah, images, single images and then same experiment again. So now I think here we have a paper already.  
And then it's in Europe. So I think it's kind of borderline because people say, well, that's text, you know, in text it's been done to them. But it's novel. We've seen by the literature survey that it's novel. So now it's April. What's the date of 87?  
27 I think we can finish images before the paper finish. Yeah, so single images and then I think it. I think we can finish images before the paper is due. Yes, and not end of May. It's on start of May.  
4th, 4th of this May. I have it in my tag here. So if we if we uh, and if the idea is to say there exists.  
Yeah, not one. Yeah, no, it was May 4th, right. So I think, yeah, if if we didn't say independent, if it's stacked or if it's an image, the geometry is the same for any fund.  
That's where we want to get to to then be able to say, well, and if we go to movie image, that's what the geometry holds as well. I think that's where we want to get to and if we can get to some mathematical axioms to say these.  
This formulation holds true whatever thing you choose. Single image video. Then I think we're on. You're you're right, you're right. We we'll we'll try to find a mathematical range of the centroid distance that holds true for text.  
Images or video. And then you say, this is it. And this explains everything. It's just the law of gravity. Yeah, it's like the law of practice. So yeah, that's. Oh, by the Newtonian gravity doesn't hold true anymore. Did you know that? I heard that. Yes, it doesn't hold true. So it's.  
It's known that it's only in gravity. So even if we call it law of gravity, some 100 years later someone will come up and say we're wrong. Yeah, wrong is not true anymore. Yeah, well, but it seems to because I think that's the the bit I really like.  
I mean, I like all of this, and it's also nice to be able to get to a. For instance, the most important finding mathematically, my whole thesis is to say that the distance between any two vertices in an N dimensional space where you normalize.  
So it holds true for L2, but not for the ball. And L2, the distance between any two points of square root of 2. That was the most important amount, which is very basic, right? If that was it. And we discussed this that night that's starting at the church, so.  
Yeah, I remember you were probably talking to me about it right outside the church while we're waiting for the pizza. We were waiting for the pizza. We integrate the chat team, the transcription, transcribing everything. Oh, yes.  
Yes, next steps. Yes, I I've written down the things that we want to try. First would be the centroid calculations and plottings, centroid overlaps, find the centroids for every class and then we can start.  
Looking at geometry, how these things overlap, what's the volume was overlapped and then we can come up with some interesting numbers. I think saying that fear overlaps most with.  
And then, but that's so and actually we have this now we have a model and we did all the whole pipeline with the model and the pipeline is there. We got what the status and say with this model with this data set we've got.  
We have these over the last four, fear and then we have two pairs, fear in this one, this is yoga, fear in the other one is yoga, and we choose and say 10 pairs, but we can't choose everything because we're going to have six. We're going to have about 13 pairs, right? We have 6 sentiments.  
7 \* 7 almost. But one thing Daniel, how do we calculate the overlap? Because it's a volume, yes, but how do we know? Like let's say I'll I'll open the image again.  
You can think about it as two circles and two deep.  
So let's say, let's say this was some something like yes and the other one was like this, but the points are so big. How do we draw a different circle that the one once we have, once we have the centroids?  
And say this is where this centroid is. And then we say and then we can imagine essentially what's the coordinate of the centroid, 00 and then we have the other one. What's the coordinate of that centroid? It's one one. And then we say, well, what's the furthest point from this centroid here?  
The furthest point from that center is at 0.40.4. What's the furthest one from this center coming around the area? And we do one circle, one on a circle in 2D. What's the area for this interesting that we will because let's say one point, even if let's say one data point with this class.  
Then if you have the and then that that's we say there's an outlier here with the outliers, this is it. But if we decrease the threshold, so 90% of where that outlier is, we still capture 99% of the data.  
OK, from central let's take, let's say we have you want to the class that is we need to talk 10 to 20% and then from those 20% of the.  
We can do it by way. That's a possibility. We can do it by density. Where is it the most dense? Yes. And then we say we want to keep the ones that are up to this density. And then it's interesting because physics gets into it. Now we hear density. That's the location for this.  
We're going to go up to density X and then screen, because then we want like 99% of reviewers. They do not look at physics. No, no, we want that because reviewers, when we see mathematical formulas, they start running, but when they see physics, they run that more fast. OK, that's amazing.  
Do you understand where this is going? If you put maths in the paper, unless it's someone like me who goes through the maths and tries to understand it, most reviewers say, yeah, this looks like an old baby. So if we have density, you'd say, well, the density.  
That that gets grabs 80% of. So you can see this is going. We can now say well we want to keep.  
The cluster not to this density, we consider that's where and then and then we do it the same with the same clusters. We say, well, what about this surprise cluster? Oh my God, the density for that if we want this density.  
With the surprise compared to the angle, let's say the angle is very tight and say look, if we keep the the the angle of this ratio, it's super tight. But to get the same density with surprise, we have to get even closer.  
Because density is so spread out, we expect some to be close here, but to match the density of OK, wait a minute.  
Let's make a goal of density. We're going to use density as I mentioned.  
So here's anger. Here's our.  
So there's a surprises here and then you will see and we'll get into the and now we look at the density of these guys.  
We'll see. Well, then that is very tightly packed. Yes. And then just something. It's now these guys, there are a little bits here, but then you say it's very small. Yes, let me say, but there are a radius of two around here. We get a certain intensity.  
And then you say, well, where do we need to have the radius around surprise? The expectation is the radius is going to be a lot of.  
Exactly. So people like this, we can say that's one approach that goes with different approaches, but we could put that in the formulation in terms of something that looks like density. It's like a problem that would also be important for this goes beyond text. This would come important in images and with you exactly.  
It goes, it kind of transcends everything. It's an abstraction. So if we're looking at this and this applies to everything and we're looking at some angle of density plus something else plus something else and this is our final formulation and it involves physics because it's geometry plus its density.  
Amazing. And working with Max.  
OK.  
That should be in here hopefully, isn't it? But also because I'm going to experiment with things. Yeah, it's here. You look at density of these guys are two of the same guys density being considered and then is it with the same busters grabs 80%? Yeah, it's all there.  
I'm going to segment that. I'm I'm tonight I'm going to time a period of density for density. Say if we have obtain a cluster of this much density for a pier, this is the radius we need.  
And so forth. But we've been working with 76 million dimensions, but it's a radius at the end of the day. It's a single distance, as many dimensions as yesterday. Doesn't even matter. It doesn't matter because it's Euclidian distance and it's like.  
Square of the sum of. Well, you get the picture, right? Amazing. I'm really excited about this. I mean, that's because of the word people in your house, like, Jesus Christ, you burned all your tokens. You know what?  
We have some money left from the.  
On the back of one thing.  
If you're interested, you turn up at the church and we'll get some EPI's for chat GDT for Codex, the government recruits work. It will burn to them overnight. Uh, on the hand. Yeah, but you can do this.  
No, The thing is, The thing is the three tokens. I'll buy the tokens and I'll spread it around. Obviously we're gonna wear most of them to the what is this up to you, you know? Yeah, no, I'll know. And for me to be to feel it up on them, I'll maintain the people there as well. Super. It's I'll, I'll help them out with whatever you taught me about.  
So that so I feel like that actually I do not because I think it's right. In fact I was uh when Nachu was said, oh I hear some of the students that we discussed this orange.  
We have no idea what's happening. Anyway, about the I just wanted to say briefly. So have we got this is a good point to pause or you want to show some other things? No, that's that's what I've done so far. OK, so I wanted to talk about the employment.  
It's not mine, it's just some bits.  
One thing that's happening, it was it was supposed to happen next week in Richmond. There's a place there called Richmond American University, London, London. And then they said, oh, we want to run an athlete and say let's run it next week and they said, OK, let's do it. But now they said.  
uh It's a bit too early. We spoke in the academics from the other departments. This is their head of department over there. I'm talking to Dinesh. And I said, come over here, me to the department. And so he said, let's organize two hackathons.  
Basically, I told him it's not costing as much because we have to get some mentors from over there to come and help the people over here. So that would be some work for you. It's not something for you to keep your minds, but it's mentorship and involves helping people do exactly what you'd like to know.  
So I'll I'll be really good. I'll be organized scientific discovery with your plums if you're interested. The other is the thing I mentioned about me, which I'm going to mention now to you, is that Narella, you might know Narella. She's a lecturer and anyway, you might have seen her before.  
She's organizing an event in July for a schoolgirl, 16 year olds, and basically there's going to be a lot of AI to them. But when you two mentioned that you were looking at all the spanning without AI, I told her and she said, yeah, that sounds like a good idea for a workshop.  
I mean, I'll, I'll, I can mention it again with you tomorrow. I'm I can listen to it now. So what what the idea of the whole thing is that we look at the project whenever we try to understand code, we don't use it.  
The AI will only create like write the script for us for the whole structure of things from like you planned out, you create, you structureize, you architecture with the architecture always and only the writing code part is done by AI.  
And you keep on challenging AI to fix the architecture. So in the sense that then you have to have a relationship where you have trust issues with AI. Does that make sense? Yeah. So you never make them do everything. You never just say yes, go on to it. You say it, tell me exactly what you're doing and you go through it.  
And then when you know that, OK, this is what you're doing, I don't know. I don't like this. I don't like that because I I'm learning about this when I was working on my new computing project. I was trying to train a CNN and transformer and this compare this for the models that I'm comparing for the project.  
And then so I was loading the data set like and I have a simple computer head, but even then it would load the complete data set like the all the data set because I didn't pick and then I have the data from local so I can just write a data loader file that would load just two images at a time whatever.  
My batch size is for the training and I only wrote that at a time. So my RAM remains, you know it doesn't pull up and my GPU also space has saved. So that's when I feel like that AI is still super. Yes, you know AI is very stupid about the simple things that you could do.  
And it just doesn't want to because what you said about context, right? If you give them enough context, if you give them enough context, they only have this amount of memory, this amount of GPU, can't go down to, how many can you load? And it says, well, I want you to load two, yes.  
So that's that you kind of have, you kind of want to have a conversation with AI to build context and then it goes and that would be a good workshop to run the girls saying that it's not about accepting the decision, it's about discussing with the AI and challenging and that would be a workshop.  
So you run a two hour thing, you get paid for it. You turn up and speak to them. They do exercises. OK, that's it. So that's gonna be July. That's each July. And then, uh, Miah was also a part of all the response. Yes. So she's in that. Yeah.  
And then, well, that's it. This is back-a-thon on Friday. We'll buy some credits, go and learn them. We progress this thing here. Oh, but I really have work on Friday. I'm more illegal another day because we'll we'll keep the money, which projects fine.  
I'll say light and cold, but I can't stay here in the morning after 10 AM. No, no problem, not stay as long as you need. And you don't have to. It doesn't have to be that day. I'm just saying because we have no, no API that you can use. That's not gonna burn any of your.  
No. So I'll definitely be there on Thursday night, but I would won't. I'll like, I'll leave around 10:00 AM on Friday, so I might not be able to be there for the presentation of hackathon, right. But I can be there in the night if that's so. OK, if you don't make, if you want to do a.  
What you call a presentation as well. And then we'll make you all the marches and for you. I don't. I don't. I don't feel like participating in that. I've already done it once something. I'm sorry. I'll make, but because I make.  
Yeah, so that I I will essentially assist you in your job there. So I feel like I'm on the sit with all the groups and make sure that or you can join a group if you want. You can help them with the project. It's up to you. You know that's because it could be the no one turns up and then we just go to the work. Yeah, and that's perfectly fine.  
I'll ask Andrew or someone to join as well. Yeah, yeah, we should. We should ask around and say, guys, we're gonna run the hack fund. We have some everything as before. Everyone's involved. There's another thing. The NHS people have been verified, unfortunately. You know, it's all people. I realise that I think we'll engage with more.  
Then, but they haven't. And so be it. Not like, yeah, yeah, yeah, maybe. I think, I think what they wanted was to use that hackathon to get to know what kind of solutions can be built and then pitch it to Snowflake. You never know.  
Potentially that's what they try to do and if to be honest, you know, good luck to them. I think they got some people out of it. They got something without me. The one thing I'm thinking about to me is and I don't, so I'm just going to say it out loud.  
I have a budget for another project with a charity. The charity they want to build. They're a music charity. They run an orchestra and they want to build an application that manages their.  
Uh, their instrument and venture. They have like an instrument library. So if you want to learn how to play the violin, you can not be joining them. We say if you're going to loan you a violin, it costs 20 lbs per month. Then you take the violin and you pay it on. So a rental thing for musical. Exactly. It's like uh.  
Inventory management thing. OK, yes, the software would be even inventory management. It's inventory management, yeah. So what I'm thinking of doing is I have to spend the money this month. I'm thinking about organizing between us like you, Andrew, George.  
And then we have some money to give out. So it's and buy a macro mini, have it run a macro mini and talk to the guy MIT here and see if we can host it so the people in the charity will have this thing online.  
Or take it's the one of them and say, well, I think why don't we just pick it as a mutual like you, me, Andrew, Josh and we guys can have just work as one team, build the whole thing. Yeah, it could be we could do it that way.  
Because I think because the hackathon, yeah, we could do it some way, just work as a team and forget about the hackathon and say it's gonna be a team team and it'll be five of us invited and then we turn on build the thing in a few hours. Yeah, we'll be able to build it in a few hours.  
I'll get the map, the Macbook meeting with the budgets and then it's like this. We can take the Macbook meeting to the lady who runs the thing like the the boss there and say this this thing.  
You'll have to sit here where you are. If you want to send the next report, you can host it over at the uni. You will have an IP address. You can access it by the way. That's what I was thinking. You can do everything you want to do in the map book.  
And basically it's it's you get the the idea right and they want AI behind it. So someone sends an e-mail and says I'd like to be part of this and I would like to play the cello. Now it's known that the cello is the most difficult instrument to borrow because they have to do this waiting list.  
What they want is that an e-mail, an AI to go through the waiting list every couple of months, e-mail those people and say, you are with you. It looks like it's going to be another eight months till you get to the top of the team. Do you still want to be able to heal? It could be a person doesn't work.  
And so do it clear. Yes, do it clear. But I think, I think we can, we can create this application. It won't be that complex because there's not a lot of AI using simple inventory management. And yeah, Andrew, Andrew has one aspect. He's good at the building the desktop. Josh.  
Knows about hardware. I think I can handle the overall architecture so we can build the team. That way we'll be able to build a lot better solution. Yeah. And if we keep, if we work separately, then I mean so you can guess another thing.  
So let's say we can build that in like a couple of hours or three hours or whatever overnight and there's still 3 hours left. There's some other projects as well that could be running on that front. There's another charity up here and then.  
And that's like they're not sure yet what they want, but I'm talking to them. I'll be talking to them next week and it's the same idea of getting these things ready, but I'm not, I'm not maybe the same. These things are AI based.  
They're running, they're online, they're doing something and it's so now it's like a what you call it, it's kind of a business. You have one man book meeting that is a server for a business, OK, but it's but also.  
We can go back to what's happening, Anita, whatever, and hit that and say guys, we put all this stuff on the map. Do you want to see it working? And then I'd say we're talking to us and operate.  
Forget about it. They might say, well, actually we need what we want to see is that they're looking instead we have a we get there, we plug it in, they see it working and then they will say, brilliant. Can we take a couple of you guys as interns? Yeah, they pick and then off you go. That's the idea.  
It turns into a sailing kind of tool. What's that sound like? That that also sounds good, but I think we'll just I'll ask them for the that thing you may or maybe you can send an e-mail or should I have not put them to go on. Do you have any chats for me?  
No, no, I was saying, I was saying we can start with the music charity. Oh, the music charities are given. That's happening. That's like 100% happening. The only decision in mind is, but now I see it's a good thing because we can use it as sales.  
Before we take them up, we need to be a charity and I can take it to them in two months and three, we can go over to the NHS and say that's I think when do you want to see how it runs? All you have to do is plug it into your network and you never have Andrew's thing.  
You're gonna have everything thing. You're gonna have everything running off that one. So on the browser, on your machine, we wanna see that running and they might say no launch it. So that's the idea of having the hardware. Yeah, that does start from.  
I don't know how within these guys will be to cover. Well, to be honest, I think the problem over there is governance. Any decision they make is so much pay or they they don't want to make decisions because the minute they say or depending on NHS would be bad after what they did just.  
Like they might just not not even respond. And then we're stuck with the manbook meeting and now with that manbook meeting, it goes to charity.  
Oh, OK, that's runs with them. That's the idea of buying. So guys, it's not even done there. I'll say this is gonna be yours on like a yearly long basis. You're welcome to use this for one year and then we can reassess. Is this being useful? Is it not being useful?  
You want to keep it, etcetera. Oh, so you're kind of saying that you wanna send hardware with hardware being in my book mini? Yeah, render it out with your software service and that it's it's well, it's software as a service because software.  
But this project is funded by UKRI. They're paying for the map for me, but they're not going to ask you that might probably be ended up for it. So it's like I owe it to almost thing, like it's like I own it. That's the annual bond. It sounds like we should if if the money is.  
I mean, we get the money for the MacBook too many anyway, so why not buy it? Yeah, I'm sure we can. We can find. Yeah, if nothing else, I'd run the models for this on it exactly. Because this clustering thing, when we move on to further N dimension, it's gonna get more expensive complication because I think that for text 768 dimensions.  
And OK, but when you move on to images.  
They're gonna be a lot more, like lot bigger vectors than just 768 sync. Yeah, so for images and and then for videos, then it's even more context that needs to be captured.  
All right, so I think that that would be this. So in short, we have some experiments around here, hopefully running on Thursday nights. And then there's the green.  
Of this workshop would be on writing without the help of AI, and then there's the things with this macro mini and the music clarity. Now we got gets going.  
So I think that's the sum total.  
Right. I'll then stop the transcription. I'll send the minutes and the transcription is going to be in the chat, but we can use that as on that. So I might throw in this thing as a LSS thing.  
What is LFS? It's a large file system for. It might need it depending on the size of the chat. So if it's like 10s of megabytes of gigs, I think I don't need it won't be 10 gigs, it's just text.  
But it will be 10 megabytes or something. It is one hour long conversation.

Sikar, Daniel** stopped transcription
