set.seed(3)

#Generate data
N=100
n=rpois(N,50)
y1=rbinom(N,n,0.6)  #CpG
y2=rbinom(N,n,0.1)  #non-CpG
y=ifelse(runif(N)<0.2, y1, y2)

#===================================


## logposterior
logpost = function(theta) {
	lambda=theta[1]
	p1=theta[2]
	p2=theta[3]
	if(lambda>=1|p1>1|p2>=1|lambda<=0|p1<=0|p2<0|p1<p2) 
            return(-999999)
        return(sum(log(lambda*p1^y*(1-p1)^(n-y) + (1-lambda)*p2^y*(1-p2)^(n-y))))
}


#===================================

proposal = function(theta) {
## jumping distribution
## choosing the standard deviation of the jumping distribution is discretionary and will affect the acceptance ratio. The ideal acceptance ratio is 10-40%. 
New=theta+rnorm(3)*c(0.01,0.01,0.01)
return(New) 
}

#===================================

NREP = 3000
## starting values
lambda = 0.2
p1 = 0.6
p2 = 0.1
mchain = data.frame(lambda=rep(NA,NREP),p1=rep(NA,NREP), p2=rep(NA,NREP))

#===================================

mchain[1,] = theta = c(lambda, p1, p2)
## keep track of acceptance rate
acc = 0; 
for(i in 2:NREP) {
## lambda
thetaCandidate = proposal(theta)
alpha = logpost(thetaCandidate)-logpost(theta)
if( runif(1) <= exp(alpha) ) {
    acc = acc+1
    theta=thetaCandidate
}
## update chain components
mchain[i,] = theta
}

#===================================

accept.ratio=acc/NREP
print(paste('Acceptance Ratio:', accept.ratio, sep=' '))
par(mfrow=c(1,3))
plot(mchain[100:NREP,1],type="l",main="lambda",ylim=0:1)
plot(mchain[100:NREP,2],type="l",main="p1",ylim=0:1)
plot(mchain[100:NREP,3],type="l",main="p2",ylim=0:1)

print(paste('Lambda Mean:', mean(mchain[100:NREP,1]), sep=' '))
print(paste('p1 Mean:', mean(mchain[100:NREP,2]), sep=' '))
print(paste('p2 Mean:', mean(mchain[100:NREP,3]), sep=' '))