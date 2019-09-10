import os
import numpy as np
import params_priors as p
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib.patches as patches

def Qlearn_multirun_tab(prior_flag):
	#this function calls the approach multiple times, and logs the resulting data
	retlog=[]
	faillog=[]
	for i in range(p.Nruns):
		print("Run no:",i)
		Q,ret,retall,pol_counts,Qpworst,failreturns=main_Qlearning_tab(prior_flag)
		if i==0:
			retlog=ret
			faillog=failreturns
		else:
			retlog=np.vstack((retlog,ret))
			faillog=np.vstack((faillog,failreturns))
		if (i+1)/p.Nruns==0.25:
			print('25% runs complete')
		elif (i+1)/p.Nruns==0.5:
			print('50% runs complete')
		elif (i+1)/p.Nruns==0.75:
			print('75% runs complete')
		elif (i+1)==p.Nruns:
			print('100% runs complete')
	meanreturns=(np.mean(retlog,axis=0))
	plt.plot(meanreturns)
	plt.show()
	return Q, retlog,retall,pol_counts,Qpworst,faillog

def select_pol(retall,tau):
	#probabilistic policy selection- PRQL step
	probs=[]
	for i in range(len(retall)):
		probs.append(np.exp(tau*retall[i]))
	probs=probs/np.sum(probs)
	sumprobs=[]
	val=0
	for i in range(len(probs)):
		val=val+probs[i]
		sumprobs.append(val)
	r=np.random.sample()
	for i in range(len(sumprobs)):
		if r<sumprobs[i]:
			pol_ID=i
		else: 
			pol_ID=len(sumprobs)-1
			break
	return pol_ID


def main_Qlearning_tab(prior_flag):
	#function to call Q learning
	Q=np.zeros((p.a,p.b,p.A))
	ld2=np.load('Qpworst_4tasks_TDerror.npy.npz')
	#Qpworst=ld2['arr_0']#start with previously learned estimate of Qp
	Qpworst=np.zeros((p.a,p.b,p.A))#uncomment this to start learning Qp from scratch
	if prior_flag==0:
		goal_state=p.targ
	else:
		goal_state=p.targ_test
	returns=[]
	failreturns=[]
	retall=np.zeros((np.shape(Qall)[0],1))
	tau=p.basetau
	use_counts=np.zeros((np.shape(Qall)[0],1))
	for i in range(p.episodes):
		pol_select=select_pol(retall,tau)
		Q,Qpworst=Qtabular_online(Q,prior_flag,pol_select,i,Qpworst)
		if (i+1)/p.episodes==0.25:
			print('25% episodes done')
		elif (i+1)/p.episodes==0.5:
			print('50% episodes done')
		elif (i+1)/p.episodes==0.75:
			print('75% episodes done')
		elif (i+1)/p.episodes==1:
			print('100% episodes done')
		if i%100==0:
			cr,failrate=calcret(Q,goal_state,i,pol_select,Qpworst)
			returns.append(cr)
			failreturns.append(failrate)
			#returns.append(ret)
		retall[pol_select]=((retall[pol_select]*use_counts[pol_select])+cr)/(use_counts[pol_select]+1)
		use_counts[pol_select]=use_counts[pol_select]+1
		tau=tau+p.deltau
		Qall[np.shape(Qall)[0]-1]=Q

	return Q, returns, retall,use_counts,Qpworst,failreturns

def Qtabular(Q,prior_flag,pol_select,episode_no,Qpworst):
	#main Q learning function
	Qp_sel=Qall[pol_select]#selected policy (PRQL)
	initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
	rounded_initial_state=staterounding(initial_state)
	if prior_flag==0:
		target_state=p.targ
	else:
		target_state=p.targ_test#change this if different target is needed when priors are used.

	while p.world[rounded_initial_state[0],rounded_initial_state[1]]==1 or np.linalg.norm(rounded_initial_state-target_state)<=p.thresh:
		initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		rounded_initial_state=staterounding(initial_state)
	state=initial_state.copy()
	roundedstate=staterounding(state)
	count=0
	breakflag=0
	reachedflag=0
	ret=0
	eps_live=1-(p.epsilon_decay*episode_no)
	ps=p.psi
	while np.linalg.norm(state-target_state)>p.thresh:
		count=count+1
		if breakflag==1:
			break
		if count>p.breakthresh:
			breakflag=1
		if pol_select==np.shape(Qall)[0]:#PRQL step
			Qmax,a=maxQ_tab(Q,state)
		else:
			if ps>np.random.sample():#PRQL step
				Qmax,a=maxQ_tab(Qp_sel,state)
			else:
				if eps_live>np.random.sample():
					#explore
					a=np.random.randint(p.A)
					if prior_flag==1:
						Qmin,aworst=minQ_tab(Qpworst,state)
						while a==aworst:
							a=np.random.randint(p.A)
				else:
					#exploit
					Qmax,a=maxQ_tab(Q,state)	

		next_state=transition(state,a)
		roundedstate=staterounding(state)
		roundednextstate=staterounding(next_state)
		#generate rewards
		if p.world[roundednextstate[0],roundednextstate[1]]==0 and next_state[0]<p.a and next_state[0]>0 and next_state[1]<p.b and next_state[1]>0:	
				if np.linalg.norm(next_state-target_state)<=p.thresh:
					R=p.highreward
					breakflag=1	
				elif np.linalg.norm(next_state-p.common_state)<=p.thresh:
					R=p.commonreward
				else:
					R=p.livingpenalty
		else: 
			R=p.penalty
			next_state=state.copy()
		#update Q values
		Qmaxnext, aoptnext=maxQ_tab(Q,next_state)
		Q[roundedstate[0],roundedstate[1],a]=Q[roundedstate[0],roundedstate[1],a]+p.alpha*(R+(p.gamma*Qmaxnext)-Q[roundedstate[0],roundedstate[1],a])
		ps=ps*p.nu#PRQL step
		state=next_state.copy()
		roundedstate=staterounding(state)
	return Q

def softmaxnorm(xorig):
	#softmax normalization
	ex=[]
	for i in range(len(xorig)):
		ex.append(np.exp(xorig[i]))
	x=ex/np.sum(ex)
	return x

def entro(Xorig):
	X=softmaxnorm(Xorig)
	XlogX=0
	for i in range(len(X)):
		XlogXi=XlogX-X[i]*np.log(X[i])
		XlogX=XlogXi
	H=(XlogX/np.log(i+1))#normalized entropy
	return H


def infer_rew(Q,state,a,next_state):
	#infer rewards from known Q function and (s,a,s')
	roundedstate=staterounding(state)
	Qmax, amax=maxQ_tab(Q,next_state)
	rew=Q[roundedstate[0],roundedstate[1],a]-p.gamma*(Qmax)
	return rew

def getQr(Q,state,act):
	#absolute advantage function
	roundedstate=staterounding(state)
	Qr=abs(Q[roundedstate[0],roundedstate[1],act]-np.max(Q[roundedstate[0],roundedstate[1],:]))
	return Qr

def checkentroflag(state,a):
	#check if conditions for state-action pair selection are satisfied
	X=[]
	for i in range(np.shape(Qall)[0]):
		Qmin, amin=minQ_tab(Qall[i],state)
		if amin==a:
			Qr=getQr(Qall[i],state,a)
			X.append(Qr)
		else:
			X.append(0)

	if entro(X)*np.mean(X)>entropy_thresh:
		entroflag=1
	else:
		entroflag=0
	return entroflag

def getpriorrew(state,a,next_state):
	#find if action is worst action
	X=[]
	for i in range(np.shape(Qall)[0]):
		Qmin, amin=minQ_tab(Qall[i],state)
		if amin==a:
			Qr=getQr(Qall[i],state,a)
			X.append(Qr)
		else:
			X.append(0)
	#Find consistency of X
	Rprior=0
	entroflag=0
	if entro(X)*np.mean(X)>entropy_thresh:
		entroflag=1
		Xnorm=softmaxnorm(X)
		for i in range(np.shape(Qall)[0]):
			Rprior=Rprior+Xnorm[i]*infer_rew(Qall[i],state,a,next_state)
	else:
		Rprior=0

	if Rprior>1:
		Rprior=1
	elif Rprior<-1:
		Rprior=-1

	return Rprior

def Qtabular_online(Q,prior_flag,pol_select,episode_no,Qpworst):
	Qp_sel=Qall[pol_select]
	initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
	rounded_initial_state=staterounding(initial_state)
	if prior_flag==0:
		target_state=p.targ
	else:
		target_state=p.targ_test

	while p.world[rounded_initial_state[0],rounded_initial_state[1]]==1 or np.linalg.norm(rounded_initial_state-target_state)<=p.thresh:
		initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		rounded_initial_state=staterounding(initial_state)
	state=initial_state.copy()
	roundedstate=staterounding(state)
	count=0
	breakflag=0
	reachedflag=0
	ret=0
	eps_live=1-(p.epsilon_decay*episode_no)
	ps=p.psi
	while np.linalg.norm(state-target_state)>p.thresh:
		count=count+1
		if breakflag==1:
			break
		if count>p.breakthresh:
			breakflag=1

		if pol_select==np.shape(Qall)[0]:
			Qmax,a=maxQ_tab(Q,state)
		else:
			if ps>np.random.sample():
				Qmax,a=maxQ_tab(Qp_sel,state)
			else:
				if eps_live>np.random.sample():
					#explore
					a=np.random.randint(p.A)
					entrfl=checkentroflag(state,a)
				else:
					#exploit
					Qmax,a=maxQ_tab(Q,state)

		next_state=transition(state,a)
		roundedstate=staterounding(state)
		roundednextstate=staterounding(next_state)

		if p.world[roundednextstate[0],roundednextstate[1]]==0 and next_state[0]<p.a and next_state[0]>0 and next_state[1]<p.b and next_state[1]>0:	
				if np.linalg.norm(next_state-target_state)<=p.thresh:
					R=p.highreward
					breakflag=1	
				elif np.linalg.norm(next_state-p.common_state)<=p.thresh:
					R=p.commonreward
				else:
					R=p.livingpenalty
		else: 
			R=p.penalty
			next_state=state.copy()
		Rprior=getpriorrew(state,a,next_state)
		Qmaxnext, aoptnext=maxQ_tab(Q,next_state)
		Qpmaxnext, apoptnext=maxQ_tab(Qpworst,next_state)
		Q[roundedstate[0],roundedstate[1],a]=Q[roundedstate[0],roundedstate[1],a]+p.alpha*(R+(p.gamma*Qmaxnext)-Q[roundedstate[0],roundedstate[1],a])
		Qpworst[roundedstate[0],roundedstate[1],a]=Qpworst[roundedstate[0],roundedstate[1],a]+p.alpha*(Rprior+(p.gamma*Qpmaxnext)-Qpworst[roundedstate[0],roundedstate[1],a])	
		ps=ps*p.nu
		state=next_state.copy()
		roundedstate=staterounding(state)
	return Q,Qpworst

def maxQ_tab(Q,state):
	Qlist=[]
	roundedstate=staterounding(state)
	for i in range(p.A):
		Qlist.append(Q[roundedstate[0],roundedstate[1],i])
	tab_maxQ=np.max(Qlist)
	maxind=[]
	for j in range(len(Qlist)):
		if tab_maxQ==Qlist[j]:
			maxind.append(j)
	if len(maxind)>1:
		optact=maxind[np.random.randint(len(maxind))]
	else:
		optact=maxind[0]
	return tab_maxQ, optact

def minQ_tab(Q,state):
	Qlist=[]
	roundedstate=staterounding(state)
	for i in range(p.A):
		Qlist.append(Q[roundedstate[0],roundedstate[1],i])
	tab_minQ=np.min(Qlist)
	minind=[]
	for j in range(len(Qlist)):
		if tab_minQ==Qlist[j]:
			minind.append(j)
	if len(minind)>1:
		optact=minind[np.random.randint(len(minind))]
	else:
		optact=minind[0]
	return tab_minQ, optact

def transition(state,act):
	n1 = np.random.uniform(low=-0.2, high=0.2, size=(1,))
	n2 = np.random.uniform(low=-0.2, high=0.2, size=(1,))
	new_state=state.copy()
	if act==0:
		new_state[0]=state[0]
		new_state[1]=state[1]+1#move up
	elif act==1:
		new_state[0]=state[0]+1#right
		new_state[1]=state[1]
	elif act==2:
		new_state[0]=state[0]
		new_state[1]=state[1]-1#move down
	elif act==3:
		new_state[0]=state[0]-1#move left
		new_state[1]=state[1]

	new_state[0]=new_state[0]+n1
	new_state[1]=new_state[1]+n2
	return new_state

def meanQ_tab(Q,state):
	Qlist=[]
	roundedstate=staterounding(state)
	for i in range(p.A):
		Qlist.append(Q[roundedstate[0],roundedstate[1],i])
	tab_maxQ=np.mean(Qlist)
	return tab_maxQ

def calcret(Q,goal_state,episode_no,pol_select,Qpworst):
	ret=0
	fails=0
	eps_live=1-(p.epsilon_decay*episode_no)
	Qp_sel=Qall[pol_select]
	for i in range(p.evalruns):
		breakflag=0
		reachedflag=0
		initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		rounded_initial_state=staterounding(initial_state)
		while p.world[rounded_initial_state[0],rounded_initial_state[1]]==1 or np.linalg.norm(rounded_initial_state-goal_state)<=p.thresh:
			initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
			rounded_initial_state=staterounding(initial_state)
		state=initial_state.copy()
		ps=p.psi
		for j in range(p.evalsteps):
			if breakflag==1:
				break
			exploreflag=0
			if pol_select==np.shape(Qall)[0]:
				Qmax,optact=maxQ_tab(Q,state)
			else:
				if ps>np.random.sample():
					Qmax,optact=maxQ_tab(Qp_sel,state)
				else:
					if eps_live>np.random.sample():
						#explore
						optact=np.random.randint(p.A)
						exploreflag=1
					else:
						#exploit
						Qmax,optact=maxQ_tab(Q,state)
			failuresflag=0
			next_state=transition(state,optact)
			roundednextstate=staterounding(next_state)
			if p.world[roundednextstate[0],roundednextstate[1]]==0 and next_state[0]<p.a and next_state[0]>0 and next_state[1]<p.b and next_state[1]>0:
				if np.linalg.norm(next_state-goal_state)<=p.thresh:
					R=p.highreward	
					breakflag=1
				elif np.linalg.norm(next_state-p.common_state)<=p.thresh:
					R=p.commonreward
				else:
					R=p.livingpenalty
			else: 
				R=p.penalty
				if exploreflag==1:
					failuresflag=1
				next_state=state.copy()
			ps=ps*p.nu
			state=next_state.copy()
			ret=ret+R*((p.gamma)**j)
			fails=fails+failuresflag
	avgsumofrew=ret/p.evalruns
	avgsumoffails=fails/p.evalruns
	return avgsumofrew,avgsumoffails

def plotmap(worldmap):
	#fig=plt.figure(0)
	for i in range(p.a):
		for j in range(p.b):
			if worldmap[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()

def staterounding(state):
	roundedstate=[0,0]
	roundedstate[0]=int(np.around(state[0]))
	roundedstate[1]=int(np.around(state[1]))
	if roundedstate[0]>(p.a-1):
		roundedstate[0]=p.a-1
	elif roundedstate[0]<0:
		roundedstate[0]=0
	if roundedstate[1]>(p.b-1):
		roundedstate[1]=p.b-1
	elif roundedstate[1]<0:
		roundedstate[1]=0
	return roundedstate

def opt_pol(Q,state,goal_state):
	plt.figure(0)
	plt.ion()
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]>0:
				plt.scatter(i,j,color='black')
	plt.show()
	pol=[]
	statelog=[]
	count=1
	while np.linalg.norm(state-goal_state)>=p.thresh:
		Qm,a=maxQ_tab(Q,state)
		
		if np.random.sample()>0.9:
			a=np.random.randint(p.A)
			
		next_state=transition(state,a)
		roundednextstate=staterounding(next_state)
		if p.world[roundednextstate[0],roundednextstate[1]]==1:
			next_state=state.copy()
		pol.append(a)
		statelog.append(state)
		print(state)
		plt.ylim(0, p.b)
		plt.xlim(0, p.a)
		plt.scatter(state[0],state[1],(60-count*0.4),color='blue')
		plt.draw()
		plt.pause(0.1)
		#input("Press [enter] to continue.")
		state=next_state.copy()
		print(count)
		if count>=100:
			break
		count=count+1
	return statelog,pol

def mapQ(Q):
	#build and view a map of Q for visualization
	plt.figure(1)
	plt.ion
	Qmap=np.zeros((p.a,p.b))
	for i in range(p.a):
		for j in range(p.b):
 			Qav=0
 			for k in range(p.A):
 				Qav=Qav+Q[i,j,k]
 			Qmap[i,j]=Qav
	Qmap=Qmap-np.min(Qmap)
	if np.max(Qmap)>0:
		Qmap=Qmap/np.max(Qmap)
	plt.imshow(np.rot90(Qmap),cmap="gray")
	plt.show()
	return Qmap

def optpol_visualize(Qp):
	#visualize the optimal policy of any Q function Qp
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]==0:
				Qmaxopt,optact=maxQ_tab(Qp,[i,j])
				if optact==0:
					plt.scatter(i,j,color='red')#up
				elif optact==1:
					plt.scatter(i,j,color='green')#right
				elif optact==2:
					plt.scatter(i,j,color='blue')#down
				elif optact==3:
					plt.scatter(i,j,color='yellow')#left

	plotmap(p.world)
	plt.show()

def optpol_visualize_min(Qp):
	#visualize the 'worst action' policy of any Q function Qp
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]==0:
				Qmaxopt,optact=minQ_tab(Qp,[i,j])
				if optact==0:
					plt.scatter(i,j,color='red')#up
				elif optact==1:
					plt.scatter(i,j,color='green')#right
				elif optact==2:
					plt.scatter(i,j,color='blue')#down
				elif optact==3:
					plt.scatter(i,j,color='yellow')#left

	plotmap(p.world)
	plt.show()

def plotmapwitharrows(points):

	fig=plt.figure(0)
	ax = plt.axes()
	axes = plt.gca()
	axes.grid(color='k')
	#QL.plt.grid(b=True, which='major', color='#666666', linestyle='-')
	axes.set_xlim([-0.5,23.5])
	axes.set_ylim([-0.5,20.5])
	xticksl=np.arange(0.5,24.5,1)#[0,1,2,3,4,5,10,15,20]
	yticksl=np.arange(0.5,21.5,1)#[0,5,10,15,20]
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.tick_params(axis=u'both', which=u'both',length=0)
	plt.xticks(xticksl)
	plt.yticks(yticksl)
	#plot arrows
	for i in range(np.shape(points)[0]):
		if points[i][2]==0:
			xpos=points[i][0]
			ypos=points[i][1]
			ax.arrow(xpos, ypos-0.4, 0, 0.5, head_width=0.5, head_length=0.3,width=0.15, fc='red', ec='red')
		elif points[i][2]==1:
			xpos=points[i][0]
			ypos=points[i][1]
			ax.arrow(xpos-0.4, ypos, 0.5, 0, head_width=0.5, head_length=0.3,width=0.15, fc='green', ec='green')
		elif points[i][2]==2:
			xpos=points[i][0]
			ypos=points[i][1]
			ax.arrow(xpos, ypos+0.4, 0, -0.5, head_width=0.5, head_length=0.3,width=0.15, fc='blue', ec='blue')
		else:
			xpos=points[i][0]
			ypos=points[i][1]
			ax.arrow(xpos+0.4, ypos, -0.5, 0, head_width=0.5, head_length=0.3,width=0.15, fc='orange', ec='orange')
	
	#plot different environments- comment/uncomment as needed-only for visualization
	#Original Env
	
	rect = patches.Rectangle((-0.50,-0.50),1,21,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((-0.50,-0.50),24,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((22.5,-0.50),1,21,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((0,19.5),23,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	#Horizontals
	rect = patches.Rectangle((-0.5,3.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((2.5,3.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((7.5,3.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,3.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((0.5,6.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((6.5,6.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((9.5,6.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)		

	rect = patches.Rectangle((0.5,10.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((6.5,9.5),10,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((6.5,12.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((10.5,12.5),6,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((0.5,15.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,15.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((9.5,15.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)


	#Verticals

	rect = patches.Rectangle((4.5,0.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,6.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,9.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,12.5),1,3,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,15.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((11.5,15.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((6.5,6.5),1,7,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((10.5,6.5),1,3,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,8.5),1,5,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,8.5),1,5,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((11.5,0.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,0.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,2.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,6.5),1,7,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,14.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,15.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,18.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((13.5,15.5),9,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,11.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,6.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((21.5,6.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	'''
	#ENVIRONMENT 2- no obst on top right
	
	rect = patches.Rectangle((-0.50,-0.50),1,21,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((-0.50,-0.50),24,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((22.5,-0.50),1,21,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((0,19.5),23,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	#Horizontals
	rect = patches.Rectangle((-0.5,3.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((2.5,3.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((7.5,3.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,3.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((0.5,6.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((6.5,6.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((9.5,6.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)		

	rect = patches.Rectangle((0.5,10.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((6.5,9.5),10,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((6.5,12.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((10.5,12.5),6,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((0.5,15.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,15.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((9.5,15.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)


	#Verticals

	rect = patches.Rectangle((4.5,0.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,6.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,9.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,12.5),1,3,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,15.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((11.5,15.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((6.5,6.5),1,7,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((10.5,6.5),1,3,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,8.5),1,5,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,8.5),1,5,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((11.5,0.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,0.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,2.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	'''
	############################################################################################
	#Environment 3-more complex
	'''
	rect = patches.Rectangle((-0.50,-0.50),1,21,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((-0.50,-0.50),24,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((22.5,-0.50),1,21,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((0,19.5),23,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	#Horizontals
	rect = patches.Rectangle((-0.5,3.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((2.5,3.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((7.5,3.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,3.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((0.5,6.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((6.5,6.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((9.5,6.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)		

	rect = patches.Rectangle((0.5,10.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((6.5,9.5),10,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((6.5,12.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((10.5,12.5),6,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((0.5,15.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,15.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((9.5,15.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)


	#Verticals

	rect = patches.Rectangle((4.5,0.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,6.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,9.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((3.5,12.5),1,3,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,15.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((11.5,15.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((6.5,6.5),1,7,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((10.5,6.5),1,3,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,8.5),1,5,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,8.5),1,5,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((11.5,0.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,0.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,2.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,6.5),1,7,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,14.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,15.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,18.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((13.5,15.5),9,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,11.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,6.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((21.5,6.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	
	rect = patches.Rectangle((17.5,8.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((8.5,0.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((19.5,1.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((8.5,17.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)		

	rect = patches.Rectangle((13.5,14.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((4.5,4.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((13.5,4.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((13.5,7.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((6.5,13.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)			

	rect = patches.Rectangle((15.5,13.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((13.5,18.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((12.5,11.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((19.5,17.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((20.5,9.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	'''
	#mark task locations
	'''
	ax.text(2.6, 1.6, '$\u03A9_{1}$',color='red', fontsize=11)
	ax.text(2.6, 17.6, '$\u03A9_{2}$',color='red', fontsize=11)
	ax.text(19.6, 17.6, '$\u03A9_{3}$',color='red', fontsize=11)
	ax.text(19.6, 1.6, '$\u03A9_{4}$',color='red', fontsize=11)
	ax.text(11.6, 8.6, '$\u03A9_{T}$',color='red', fontsize=11)
	'''

	#ax.text(11.5, 8.5, '$\u03A9^{\'}_{T}$',color='red', fontsize=11)


def loadtasks():
	#load either original, or one of the modified environments. Also change params file accordingly
	t1=np.load("task1_final.npy.npz")
	t2=np.load("task2_final.npy.npz")
	t3=np.load("task3_final.npy.npz")
	t4=np.load("task4_final.npy.npz")
	t5=np.load("task5_final.npy.npz")
	t6=np.load("task6_final.npy.npz")
	t7=np.load("task7_final.npy.npz")
	t8=np.load("task8_final.npy.npz")
	t9=np.load("task9_final.npy.npz")
	t10=np.load("task10_final.npy.npz")
	t1mod=np.load("task1_mod_final.npy.npz")
	t2mod=np.load("task2_mod_final.npy.npz")
	t3mod=np.load("task3_mod_final.npy.npz")
	t4mod=np.load("task4_mod_final.npy.npz")
	t5mod=np.load("task5_mod_final.npy.npz")
	t6mod=np.load("task6_mod_final.npy.npz")
	t7mod=np.load("task7_mod_final.npy.npz")
	t8mod=np.load("task8_mod_final.npy.npz")
	t9mod=np.load("task9_mod_final.npy.npz")
	t10mod=np.load("task10_mod_final.npy.npz")
	t1com=np.load("task1_comrew_final.npy.npz")
	t2com=np.load("task2_comrew_final.npy.npz")
	t3com=np.load("task3_comrew_final.npy.npz")
	t4com=np.load("task4_comrew_final.npy.npz")
	t5com=np.load("task5_comrew_final.npy.npz")
	t6com=np.load("task6_comrew_final.npy.npz")
	t7com=np.load("task7_comrew_final.npy.npz")
	t8com=np.load("task8_comrew_final.npy.npz")
	t9com=np.load("task9_comrew_final.npy.npz")
	t10com=np.load("task10_comrew_final.npy.npz")
	t1complex=np.load("complexmod_task1.npy.npz")
	#t2complex=np.load("complexmod_task2.npy.npz")
	t3complex=np.load("complexmod_task3.npy.npz")
	t4complex=np.load("complexmod_task4.npy.npz")


	#Qvalues
	Q_t1=t1['arr_0']
	Q_t2=t2['arr_0']
	Q_t3=t3['arr_0']
	Q_t4=t4['arr_0']
	Q_t5=t5['arr_0']
	Q_t6=t6['arr_0']
	Q_t7=t7['arr_0']
	Q_t8=t8['arr_0']
	Q_t9=t9['arr_0']
	Q_t10=t10['arr_0']
	Q_t1mod=t1mod['arr_0']
	Q_t2mod=t2mod['arr_0']
	Q_t3mod=t3mod['arr_0']
	Q_t4mod=t4mod['arr_0']
	Q_t5mod=t5mod['arr_0']
	Q_t6mod=t6mod['arr_0']
	Q_t7mod=t7mod['arr_0']
	Q_t8mod=t8mod['arr_0']
	Q_t9mod=t9mod['arr_0']
	Q_t10mod=t10mod['arr_0']
	Q_t1com=t1com['arr_0']
	Q_t2com=t2com['arr_0']
	Q_t3com=t3com['arr_0']
	Q_t4com=t4com['arr_0']
	Q_t5com=t5com['arr_0']
	Q_t6com=t6com['arr_0']
	Q_t7com=t7com['arr_0']
	Q_t8com=t8com['arr_0']
	Q_t9com=t9com['arr_0']
	Q_t10com=t10com['arr_0']
	Q_t1complex=t1complex['arr_0']
	#Q_t2complex=t2complex['arr_0']
	Q_t3complex=t3complex['arr_0']
	Q_t4complex=t4complex['arr_0']

	#Qall=[Q_t1complex,Q_t3complex,Q_t4complex]#,Q_t5,Q_t6,Q_t7,Q_t8,Q_t9,Q_t10]
	#Qall=[Q_t1mod,Q_t2mod,Q_t3mod,Q_t4mod]#,Q_t5,Q_t6,Q_t7,Q_t8,Q_t9,Q_t10]
	#Qall=[Qt1mod,Qt2mod,Qt3mod,Qt4mod,Qt5mod,Qt6mod,Qt7mod,Qt8mod,Qt9mod,Qt10mod]
	Qall=[Q_t1,Q_t2,Q_t3,Q_t4]
	return Qall
#######################################
prior_flag=1
#ld=np.load('priors_decayeps_sam100.npy.npz')
ld=np.load('Qpbest_4tasks_TDerror.npy.npz')
Qpprior=ld['arr_0']
ld2=np.load('Qpworst_4tasks_TDerror.npy.npz')
#ld2=np.load('newQppriors.npy.npz')#-alternative
Qpworst=ld2['arr_0']
###########################
#assemble previous Q functions
Qall=loadtasks()
if __name__=="__main__":
	points=[]
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]==0:
				for k in range(p.A):
					entrfl=0
					entrfl=checkentroflag([i,j],k)
					if entrfl==1:
						if np.shape(points)[0]==0:
							points=np.array([i,j,k])
						else:
							points=np.vstack((points,np.array([i,j,k])))

	plotmapwitharrows(points)
	#plotmap(p.world)
	plt.show()

	Q,retlog,retall,pol_counts,Qpworst,faillog=Qlearn_multirun_tab(prior_flag)
	mr=(np.mean(retlog,axis=0))
	csr=[]
	for i in range(len(mr)):
		if i>0:			
			csr.append(np.sum(mr[0:i])/i)
	plt.figure(2)
	plt.plot(csr)
	plt.show()

######optimal policy#####
#statelog,pol=opt_pol(w,np.array([15.,25.]))