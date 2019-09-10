import tabularQlearning as QL
import numpy as np
import params_priors as p
import time
import matplotlib.patches as patches


def QrfromQ(state,Q):
	#advantage function
	roundedstate=QL.staterounding(state)
	Qr=Q[roundedstate[0],roundedstate[1],:]
	Qr=Qr-np.max(Qr)
	return Qr

def sample_states(states):
	#only used in offline setting- sample state action pairs
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]==0:
				if np.shape(states)[0]==0:
					states=np.array([i,j])
				else:
					states=np.vstack((states,np.array([i,j])))
		

	return states

def bwfromQ(state,Q):
	maxQval,opta=QL.maxQ_tab(Q,state)
	minQval,nonopt=QL.minQ_tab(Q,state)
	Qr=QrfromQ(state,Q)
	for j in range(p.A):
		#best actions-used only in supplementary material
		if j==0:
			if j==opta:
				b=abs(Qr[j])
			else:b=0
		elif j==opta:
			b=np.vstack((b,abs(Qr[j])))
		else:b=np.vstack((b,0))
		#worst actions
		if j==0:
			if j==nonopt:
				w=abs(Qr[j])
			else:w=0
		elif j==nonopt:
			w=np.vstack((w,abs(Qr[j])))
		else:w=np.vstack((w,0))
	return b,w

def BW(state,Qall):
	B=[]
	W=[]
	siz_Qall=np.shape(Qall)
	for i in range(siz_Qall[0]):
		b,w=bwfromQ(state,Qall[i])
		if i==0:
			B=b
			W=w
		else:
			B=np.hstack((B,b))#for best actions - typically not used
			W=np.hstack((W,w))#for worst actions
	return B,W

def softmaxnorm(xorig):
	#softmax normalization
	ex=[]
	for i in range(len(xorig)):
		ex.append(np.exp(xorig[i]))
	x=ex/np.sum(ex)
	return x

def entro(X):
	sizX=np.shape(X)
	H=np.zeros((sizX[0],1))
	for j in range(sizX[0]):
		xorig=X[j]
		x=softmaxnorm(xorig)
		plogp=0
		for i in range(len(x)):
			plogpi=plogp-x[i]*np.log(x[i])
			plogp=plogpi	
		entr=(plogp/np.log(i+1))#normalized entropy
		H[j]=np.mean(xorig)*entr#normalized entropy*mean of weights
	#H is computed across tasks, for each action
	maxH=np.max(H)
	maxinds=[]
	for i in range(sizX[0]):
		if maxH==H[i]:
			if len(maxinds)==0:
				maxinds=[i]
			else: 
				maxinds.append(i)
	if len(maxinds)>1:
		act=maxinds[np.random.randint(len(maxinds))]
	else:
		act=maxinds[0]
	#act is the action for which H*W is maximum(least variation across tasks)
	return H, act

def prior_rewards(rstates,state,next_state,a,Qall):
	#R=0
	roundedstate=QL.staterounding(state)
	roundednextstate=QL.staterounding(next_state)
	#worst state action pairs
	
	R=0
	#best state action pairs
	if np.shape(rstates)[0]>3:
		for i in range(np.shape(rstates)[0]):
			roundedrstate=QL.staterounding(rstates[i][0])
			if np.linalg.norm(state-rstates[i][0])<p.thresh and a==rstates[i][1]:
			#if roundedstate[0]==roundedrstate[0] and roundedstate[1]==roundedrstate[1] and a==rstates[i][1]:
				bQ=0
				X=[]
				for j in range(len(rstates[i][2])):
					X.append(rstates[i][2][j])
				Xnorm=softmaxnorm(X)
				for j in range(len(rstates[i][2])):
					Qmaxnext,astar=QL.maxQ_tab(Qall[j],next_state)
					deltaQ=Qall[j][roundedstate[0],roundedstate[1],a]-p.gamma*Qmaxnext
					bQ=bQ+Xnorm[j]*deltaQ
				R=bQ#/np.sum(rstates[i][2])

	if R>1:
		R=1
	if R<-1:
		R=-1

	return R

def viz_rewards(rstates,Qall):
	rmap=np.zeros((p.a,p.b,p.A))
	for i in range(p.a):
		for j in range(p.b):
			for k in range(p.A):
				next_state=QL.transition(np.array([i,j]),k)
				rmap[i,j,k]=prior_rewards(rstates,np.array([i,j]),next_state,k,Qall)
				#print(prior_rewards(pstates,rstates,state,next_state,k,Qall))
	return rmap

def maprewards(rewards):
	rewards=rewards-np.min(rewards)
	if np.max(rewards)>0:
		rewards=rewards/np.max(rewards)
	QL.plt.imshow(np.rot90(rewards),cmap="gray")
	QL.plt.show()

def Qtabular_priors(Q,pstates,rstates,episode_no):
	initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
	rounded_initial_state=QL.staterounding(initial_state)
	while p.world[rounded_initial_state[0],rounded_initial_state[1]]==1:
		initial_state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		rounded_initial_state=QL.staterounding(initial_state)
	state=initial_state.copy()
	count=0
	breakflag=0
	eps_live=1-(p.epsilon_decay*episode_no)
	for ii in range(p.breakthresh):
		count=count+1
		if breakflag==1:
			break

		#if eps_live>np.random.sample():#decaying epsilon- alternative
		if p.epsilon>np.random.sample():
			#explore
			a=np.random.randint(p.A)
		else:
			#exploit
			#a=np.random.randint(p.A)
			Qmax,a=QL.maxQ_tab(Q,state)

		next_state=QL.transition(state,a)
		roundednextstate=QL.staterounding(next_state)
		roundedstate=QL.staterounding(state)

		R=prior_rewards(pstates,state,next_state,a,Qall)

		if p.world[roundednextstate[0],roundednextstate[1]]==1 or next_state[0]>p.a or next_state[0]<0 or next_state[1]>p.b or next_state[1]<0:	
			next_state=state.copy()
		Qmaxnext, aoptnext=QL.maxQ_tab(Q,next_state)
		#if R!=0:
		Q[roundedstate[0],roundedstate[1],a]=Q[roundedstate[0],roundedstate[1],a]+p.alpha*(R+(p.gamma*Qmaxnext)-Q[roundedstate[0],roundedstate[1],a])

		state=next_state.copy()
	return Q

def main_Qlearning_priors(rstates,pstates,Q):
	if np.shape(Q)[0]==0:
		Q=np.zeros((p.a,p.b,p.A))
		print('blah')
	else: 
		#ld=np.load('priors_onlyisland_withinit.npy.npz')
		ld=np.load('newQppriors.npy.npz')#previously learned Qp
		Q=ld['arr_0'].copy()
	returns=[]
	for i in range(p.episodes):
		print(i)
		#if i%100==0:
		#	QL.mapQ(Q)
		if (i+1)/p.episodes==0.25:
			print('25% episodes done')
		elif (i+1)/p.episodes==0.5:
			print('50% episodes done')
		elif (i+1)/p.episodes==0.75:
			print('75% episodes done')
		elif (i+1)/p.episodes==1:
			print('100% episodes done')
		Q=Qtabular_priors(Q,pstates,rstates,i)
		if i%1==0:
			returns.append(calcret(Q,pstates,i))
		if len(returns)>2 and abs(returns[len(returns)-1])<p.Qthresh and abs(returns[len(returns)-2])<p.Qthresh:
			print(i)
			#break
		else:
			print(abs(returns[len(returns)-1]))
	return Q,returns

def calcret(Q,rstates,episode_no):
	#function to compute performance of agent after each episode. 
	ret=0
	TD=0
	eps_live=1-(p.epsilon_decay*episode_no)
	for i in range(p.evalruns):
		state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
		roundedstate=QL.staterounding(state)
		while p.world[roundedstate[0],roundedstate[1]]==1:
			state=np.array([(p.a-1)*np.random.random_sample(), (p.b-1)*np.random.random_sample()])
			roundedstate=QL.staterounding(state)
		breakflag=0
		for j in range(p.evalsteps):
			if breakflag==1:
				break
			
			#if eps_live>np.random.sample():#alternative for decaying epsilon
			if p.epsilon>np.random.sample():
				optact=np.random.randint(p.A)			
			else:
				Qmaxopt,optact=QL.maxQ_tab(Q,state)
			
			next_state=QL.transition(state,optact)
			roundednextstate=QL.staterounding(next_state)

			if p.world[roundednextstate[0],roundednextstate[1]]==0 and next_state[0]<p.a and next_state[0]>0 and next_state[1]<p.b and next_state[1]>0:
				R=prior_rewards(rstates,state,next_state,optact,Qall)
			else: 
				R=prior_rewards(rstates,state,next_state,optact,Qall)
				next_state=state.copy()
				#TD=TD+1
			round_state=QL.staterounding(state)
			Qmaxopteval,optacteval=QL.maxQ_tab(Q,state)
			tderr=R+p.gamma*Qmaxopteval-Q[round_state[0],round_state[1],optact]
			state=next_state.copy()
			ret=ret+R*(p.gamma**j)
			TD=TD+abs(tderr)
	avgsumofrew=ret/p.evalruns
	#avgsumofrew=TD/p.evalruns#uncomment this to replace return with TD error
	return avgsumofrew

def plotmapwitharrows(points):

	fig=QL.plt.figure(0)
	ax = QL.plt.axes()
	axes = QL.plt.gca()
	axes.grid(color='k')
	#QL.plt.grid(b=True, which='major', color='#666666', linestyle='-')
	axes.set_xlim([-0.5,23.5])
	axes.set_ylim([-0.5,20.5])
	xticksl=np.arange(0.5,24.5,1)#[0,1,2,3,4,5,10,15,20]
	yticksl=np.arange(0.5,21.5,1)#[0,5,10,15,20]
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.tick_params(axis=u'both', which=u'both',length=0)
	QL.plt.xticks(xticksl)
	QL.plt.yticks(yticksl)
	
	for i in range(np.shape(points)[0]):
		if points[i][1]==0:
			xpos=points[i][0][0]
			ypos=points[i][0][1]
			#if xpos!=14 and ypos!=20:
			ax.arrow(xpos, ypos-0.4, 0, 0.5, head_width=0.5, head_length=0.3,width=0.15, fc='red', ec='red')
			#QL.plt.scatter(sel_s_a_W[i][0][0],sel_s_a_W[i][0][1],color='red')
		elif points[i][1]==1:
			xpos=points[i][0][0]
			ypos=points[i][0][1]
			ax.arrow(xpos-0.4, ypos, 0.5, 0, head_width=0.5, head_length=0.3,width=0.15, fc='green', ec='green')
			#QL.plt.scatter(sel_s_a_W[i][0][0],sel_s_a_W[i][0][1],color='green')
		elif points[i][1]==2:
			xpos=points[i][0][0]
			ypos=points[i][0][1]
			ax.arrow(xpos, ypos+0.4, 0, -0.5, head_width=0.5, head_length=0.3,width=0.15, fc='blue', ec='blue')
			#QL.plt.scatter(sel_s_a_W[i][0][0],sel_s_a_W[i][0][1],color='blue')
		else:
			xpos=points[i][0][0]
			ypos=points[i][0][1]
			ax.arrow(xpos+0.4, ypos, -0.5, 0, head_width=0.5, head_length=0.3,width=0.15, fc='orange', ec='orange')
			#QL.plt.scatter(sel_s_a_W[i][0][0],sel_s_a_W[i][0][1],color='yellow')
	

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
	
	####################
	
	###Original Env#######
	
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
	

	
	########################
	#comment to get mod ENVIRONMENT 
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
	
	
	
	####################
	'''
	###Complex Env#######

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

	
		####################
	'''
	###only island Env#######
	
	#Horizontals

	rect = patches.Rectangle((6.5,6.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((9.5,6.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
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

	#Verticals

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
	'''
	
	'''
	############
	##noisland environment
	######

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

	rect = patches.Rectangle((0.5,10.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
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


	rect = patches.Rectangle((11.5,0.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,0.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((16.5,2.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	########################
	#comment to get ENVIRONMENT 1
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
	'''
	####################
	###Shifted Env#######
	#Horizontals
	rect = patches.Rectangle((2.5,5.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((4.5,5.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((9.5,5.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((17.5,5.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((2.5,8.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((8.5,8.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((11.5,8.5),7,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)		

	rect = patches.Rectangle((2.5,12.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)	

	rect = patches.Rectangle((8.5,11.5),10,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((8.5,14.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((12.5,14.5),6,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((2.5,17.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((7.5,17.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((11.5,17.5),3,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)


	#Verticals

	rect = patches.Rectangle((6.5,2.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,8.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,11.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((5.5,14.5),1,3,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((7.5,17.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((13.5,17.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((8.5,8.5),1,7,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((12.5,8.5),1,3,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((17.5,10.5),1,5,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((17.5,10.5),1,5,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((13.5,2.5),1,4,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,2.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,4.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	########################
	#comment to get ENVIRONMENT 1
	rect = patches.Rectangle((20.5,8.5),1,7,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((20.5,16.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,17.5),1,2,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((18.5,20.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((15.5,17.5),9,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((20.5,13.5),4,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((20.5,8.5),2,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((23.5,8.5),1,1,linewidth=0.1,edgecolor='black',facecolor='black')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((19.5,17.5),1,1,linewidth=0.1,edgecolor='white',facecolor='white')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)

	rect = patches.Rectangle((21.5,8.5),1,1,linewidth=0.1,edgecolor='white',facecolor='white')
	ax = fig.add_subplot(111)
	ax.add_patch(rect)
	'''

	QL.plt.rcParams['font.weight']= 'bold'
	ax.text(2.5, 1.6, '$\u03A9_{1}$',color='red', fontsize=11)
	ax.text(2.5, 17.6, '$\u03A9_{2}$',color='red', fontsize=11)
	ax.text(19.5, 17.6, '$\u03A9_{3}$',color='red', fontsize=11)
	ax.text(19.5, 1.6, '$\u03A9_{4}$',color='red', fontsize=11)
	ax.text(11.5, 8.6, '$\u03A9_{T}$',color='red', fontsize=11)
	

	'''
	ax.text(7.5, 1.5, '$\u03A9^{\'}_{1}$',color='red', fontsize=20)
	ax.text(9.5, 17.5, '$\u03A9^{\'}_{2}$',color='red', fontsize=20)
	ax.text(15.5, 13.5, '$\u03A9^{\'}_{3}$',color='red', fontsize=20)
	ax.text(13.5, 1.5, '$\u03A9^{\'}_{4}$',color='red', fontsize=20)
	'''

	#ax.text(11.5, 8.5, '$\u03A9^{\'}_{T}$',color='red', fontsize=11)

	#fig.savefig('trial.tif')

def optpol_visualize(Qp):
	for i in range(p.a):
		for j in range(p.b):
			if p.world[i,j]==0:
				Qmaxopt,optact=QL.maxQ_tab(Qp,[i,j])
				if optact==0:
					QL.plt.scatter(i,j,color='red')
				elif optact==1:
					QL.plt.scatter(i,j,color='green')
				elif optact==2:
					QL.plt.scatter(i,j,color='blue')
				elif optact==3:
					QL.plt.scatter(i,j,color='orange')

	QL.plotmap(p.world)
	QL.plt.show()


def loadtasks():
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

	
	t1_noisland=np.load("task1_noisland.npy.npz")
	t2_noisland=np.load("task2_noisland.npy.npz")
	t3_noisland=np.load("task3_noisland.npy.npz")
	t4_noisland=np.load("task4_noisland.npy.npz")

	Q_t1_noisland=t1_noisland['arr_0']
	Q_t2_noisland=t2_noisland['arr_0']
	Q_t3_noisland=t3_noisland['arr_0']
	Q_t4_noisland=t4_noisland['arr_0']
	
	
	t1_onlyisland=np.load("task1_onlyisland.npy.npz")
	t2_onlyisland=np.load("task2_onlyisland.npy.npz")
	t3_onlyisland=np.load("task3_onlyisland.npy.npz")
	t4_onlyisland=np.load("task4_onlyisland.npy.npz")

	Q_t1_onlyisland=t1_onlyisland['arr_0']
	Q_t2_onlyisland=t2_onlyisland['arr_0']
	Q_t3_onlyisland=t3_onlyisland['arr_0']
	Q_t4_onlyisland=t4_onlyisland['arr_0']
	
	t1_shiftedenv=np.load("task1_shiftedenv.npy.npz")
	t2_shiftedenv=np.load("task2_shiftedenv.npy.npz")
	t3_shiftedenv=np.load("task3_shiftedenv.npy.npz")
	t4_shiftedenv=np.load("task4_shiftedenv.npy.npz")

	Q_t1_shiftedenv=t1_shiftedenv['arr_0']
	Q_t2_shiftedenv=t2_shiftedenv['arr_0']
	Q_t3_shiftedenv=t3_shiftedenv['arr_0']
	Q_t4_shiftedenv=t4_shiftedenv['arr_0']

	#Qall=[Q_t1_shiftedenv,Q_t2_shiftedenv,Q_t3_shiftedenv,Q_t4_shiftedenv]
	#Qall=[Q_t1_onlyisland,Q_t2_onlyisland,Q_t3_onlyisland,Q_t4_onlyisland]
	#Qall=[Q_t1complex,Q_t3complex,Q_t4complex]#,Q_t5,Q_t6,Q_t7,Q_t8,Q_t9,Q_t10]
	#Qall=[Q_t1_noisland,Q_t2_noisland,Q_t3_noisland,Q_t4_noisland]
	#Qall=[Q_t1mod,Q_t2mod,Q_t3mod,Q_t4mod]#,Q_t5,Q_t6,Q_t7,Q_t8,Q_t9,Q_t10]
	#Qall=[Qt1mod,Qt2mod,Qt3mod,Qt4mod,Qt5mod,Qt6mod,Qt7mod,Qt8mod,Qt9mod,Qt10mod]
	Qall=[Q_t1,Q_t2,Q_t3,Q_t4]
	return Qall


if __name__=="__main__":
	###########################
	#assemble previous Q functions
	Qall=loadtasks()
	print(np.shape(Qall))
	#sample Ns states
	states=[]
	states=sample_states(states)
	state=states[0]
	############################

	#get B(s,a) and W(s,a) across tasks for each sampled state
	#compute H(B(s,a)) and H(W(s,a))

	HBlog=0
	HWlog=0
	sel_s_a_B=[]
	sel_s_a_W=[]
	for j in range(np.shape(states)[0]):
		state=states[j]
		B,W=BW(state,Qall)
		HB,act_b=entro(B)
		HW,act_w=entro(W)

	#select a subset of (s,a) pairs if H(B) or H(W) >t

		if np.max(HB)>=0.5:
			if np.shape(sel_s_a_B)[0]==0:
				sel_s_a_B=[state,act_b,B[act_b]]
			else:
				sel_s_a_B=np.vstack((sel_s_a_B,[state,act_b,B[act_b]]))

		if np.max(HW)>=0.5:#p.entropy_thresh:
			if np.shape(sel_s_a_W)[0]==0:
				sel_s_a_W=[state,act_w,W[act_w]]
			else:
				sel_s_a_W=np.vstack((sel_s_a_W,[state,act_w,W[act_w]]))

	print("W:",sel_s_a_W)
	#print("B:",sel_s_a_B)
	plotmapwitharrows(sel_s_a_W)
	#QL.plotmap(p.world)
	QL.plt.show()


	rmap=viz_rewards(sel_s_a_W,Qall)

	for i in range(p.Nruns):
		#ld=np.load('Qpworst_4tasks_TDerror.npy.npz')
		ld=np.load('newQppriors.npy.npz')
		Qp=ld['arr_0']
		#Qp=[]
		Qp,returns=main_Qlearning_priors(sel_s_a_B,sel_s_a_W,Qp)
		if i==0:
			returnlog=returns
		else:
			returnlog=np.vstack((returnlog,returns))


	#Qpp=Qp.copy()

	#Qt,retlogt,,retall,pol_counts=QL.Qlearn_multirun_tab(1,Qpp)#