**Start HPC Experiments-20260412_204326-Meeting Transcript**

12 April 2026, 7:43pm

6m 36s

Sikar, Daniel** started transcription

PG-Verma, Pritish Ranjan** 0:03  
Me to, yes, so for this model training experiment, what I've done is inside our repo vid IQ HPC.  
So, yeah, same signs, yeah, carry on.  
Inside vid IQ HPC repo, inside the experiments where we're doing most of our experiments, there was text and then embedding field experiments. I've added the text, a text model folder inside which.  
We have a trained multi-class model P.Y. This is the this is the this is the P.Y. file that has the this architecture that we were talking about, right? Where there's transformer, then 768 embedding size, right? Fully connected layers that gives you the logits and then softmax after it.  
The model that I've used right now here is quen 3 1.7 billion parameters, right? Discussed that 600 million is a little less, 1.7 billion would be a good starting point, right? OK.  
I don't yet know how much VRAM we would need to train it, but if it's 1.7 billion, I assume... It would be about three times more, sometimes 4, so maybe 8 would be safe, 8 gigs, which we have on the card. So we're good. Wonderful. And then... So we have 40 gigs and 80 gigs card, so that should train easily on us. Yes.  
So I've also I've also put in a readme file for context when you shift to when you work on your PC. So the folder we're looking at here just for the for the transcript is in vid IQ HPC. We have a.  
Experiments, in experiments we have a test model, in test model we have a scripts. No, we actually don't. Let me put the scripts there.  
Oh.  
Oh no, I've already pushed it, so let's not make any changes anymore. This script just added later. This is just to philtre the data set. I've already filtered it. Right. So we can delete it with this. So we're looking at the five classes. Yes, we're looking at the five classes that we had.  
Right. Joy is now happiness, but I think it's the same thing because in the new data set, the joy label is labelled as happiness. Okay. That is fine. Everything else remains same. So when on HPC, you would have all the code. If you have at the time, then you can order on HPC or we can do it later on, but essentially the idea is to train.  
The model on the data set that I just gave you, and then once the model is trained, then we look at the embeddings at this layer and this layer, so this is here we're going to get Nguyen get rid of that bit and add this up over doing.  
All of this is Nguyen, right? This is Nguyen. Nguyen, we have taken the layer that the Nguyen has output layers and we have picked the one that is at 768 embedding size, right? And then from there we get the logic.  
OK, this is done by fully connected layers, so just some normal neuron multiplication, alright, and then, and then when essentially our idea is to be looking at this and this once the training is completed, we look, we do the same set of experiments on.  
That we did earlier on pre-trained model, but assuming that this model, let's see how this and you push that to the vid IQ vid IQ HPC. Yes, it's there. Super. It's in experiments and text model. Alright, the three files requirement, read me, and...  
Train multi-class requirements. I think it has created on its own, but it's good that it's there. Yeah, now you'd be able to create the right environment for training the model. Alright, OK. That that's all that that's all that. Once the training is completed, we can move on to. Alright, super. Excellent.  
Okay.  
Do you think you have the HPC access to it right now? So I have the access and I'm configuring it. The thing is, the access at the moment, I have to go through a Windows machine.  
So...  
But it's all there. And now I'm going to run an experiment, make sure I can run something on the HPC, and then I can clone this repo and start running that, this training. Okay. I mean, if you can, then it's okay. Otherwise, we'll figure out some other time. Yeah.  
But it means I'll probably going to add an HPC folder to that with the batch scripts to make it work on the HPC. No, you can change the script any way as long as you ensure that the model architecture is there. I mean, I don't mind if you even wish to change Gwen 3.  
into some other model. Well, it's not that one. Yes, I mean, you can change anything. That's why I've created the read me file. So as soon as you go to your cloud, you just ask it to read this so it has some context of what we're trying to do. And then you can change the model as you wish. Essentially, the only two important thing is that we still have an embedding that is in 768 size somewhere.  
In the model that we can and that we can retrieve and the fully connect and the five size logits, which I think any model you train will have the output shape as five, because we're dealing with five emotions, so that part's OK, but you can change anything as you wish, because this is this is a completely new set of experiments, so we're not.  
We can start anywhere with anything. All right.  
Alright.  
I think that's about it. I'll take a leave, I'll go home, I'll go for dinner.  
Let me stop the transcript.

Sikar, Daniel** stopped transcription
