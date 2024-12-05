a=read.csv("C:\\Users\\Tuur Willio\\Desktop\\bachelor proof\\layer1_9.txt", header = FALSE, sep = ";", dec = ".")
a=a[,1]

b=read.csv("C:\\Users\\Tuur Willio\\Desktop\\bachelor proof\\layer2_9.txt", header = FALSE, sep = ";", dec = ".")
b=b[,1]


data2 = read.csv("C:\\Users\\Tuur Willio\\Desktop\\bachelor proof\\DATA_tensor_r^2~data.txt", header = FALSE, sep = ";", dec = ".")
data2 = data2[-10,]
plot(seq(from = 10, to = 500, by = 10), data2[1,2:51])
lines(seq(from = 10, to = 500, by = 10), data2[1,2:51])
colM = colMeans(data2)[2:51]
plot(seq(from = 10, to = 500, by = 10), colM)
lines(seq(from = 10, to = 500, by = 10), colM)

#for layout ------------------
data = read.csv("C:\\Users\\Tuur Willio\\Desktop\\bachelor proof\\DATA_r^2_congif.txt", header = FALSE, sep = ";", dec = ".")
library(reshape2)
library(multcomp)
library(dplyr)
library(car)
a=melt(data, id.vars = "V1")
a[colnames(a)=="variable"]=NULL
colnames(a)[1] = "Conf"
a[,1] = factor(a[,1])
str(a)

m=aov(value~Conf, data=a)

leveneTest(value~Conf, data=a)
bartlett.test(value~Conf, data = a)

oneway.test(value ~ Conf, data = a, var.equal = FALSE)

pairwise.t.test(a$value, a$Conf, p.adjust.method="bonferroni", var.equal=FALSE)

a$Conf =  recode_factor(a$Conf, "[1, 25, 25, 25, 25, 1]" = "[1, (25,)^4, 1]","[1, 25, 25, 25, 25, 25, 25, 25, 25, 1]" = "[1, (25,)^8, 1]")

vioplot(value~Conf, data=a)
boxplot(value~Conf, data=a,cex.lab=2, cex.axis=2, cex=2)
m.mcp = glht(m,linfct=mcp(Conf="Tukey"))
summary(m.mcp, test=adjusted("bonferroni"))



t.test(a,b,paired = TRUE, alternative = "two.sided")
library(vioplot)

