graphics.off(); rm(list=ls(all=TRUE))
library(caret); library(rpart);library(fields)

train=read.csv("../input/train.csv")
weather=read.csv("../input/weather.csv")
spray=read.csv("../input/spray.csv")
test=read.csv("../input/test.csv")

# deal with missing values in weather ======================================
# Tavg
weather$Tavg=as.character(weather$Tavg)
for(i in which(weather$Tavg=="M")){
  if(i%%2==0){weather$Tavg[i]=weather$Tavg[i-1]}
  else{weather$Tavg[i]=weather$Tavg[i+1]}}
weather$Tavg=as.numeric(weather$Tavg)
# Depart
weather$Depart=as.character(weather$Depart)
for(i in which(weather$Depart=="M")){
  if(i%%2==0){weather$Depart[i]=weather$Depart[i-1]}
  else{weather$Depart[i]=weather$Depart[i+1]}}
weather$Depart=as.numeric(weather$Depart)
# DewPoint
weather$DewPoint=as.character(weather$DewPoint)
weather$DewPoint=as.numeric(weather$DewPoint)
# WetBulb
weather$WetBulb=as.character(weather$WetBulb)
for(i in which(weather$WetBulb=="M")){
  if(i%%2==0){weather$WetBulb[i]=weather$WetBulb[i-1]}
  else{weather$WetBulb[i]=weather$WetBulb[i+1]}}
weather$WetBulb=as.numeric(weather$WetBulb)
weather$WetBulb=weather$WetBulb-weather$Tavg
# Heat
weather$Heat=as.character(weather$Heat)
for(i in which(weather$Heat=="M")){
  if(i%%2==0){weather$Heat[i]=weather$Heat[i-1]}
  else{weather$Heat[i]=weather$Heat[i+1]}}
weather$Heat=as.numeric(weather$Heat)
# Cool
weather$Cool=as.character(weather$Cool)
for(i in which(weather$Cool=="M")){
  if(i%%2==0){weather$Cool[i]=weather$Cool[i-1]}
  else{weather$Cool[i]=weather$Cool[i+1]}}
weather$Cool=as.numeric(weather$Cool)

# Sunrise
weather$Sunrise=as.character(weather$Sunrise)
for(i in which(weather$Sunrise=="-")){
  weather$Sunrise[i]=weather$Sunrise[i-1]}
weather$hSunrise=substr(weather$Sunrise,1,2)
weather$mSunrise=substr(weather$Sunrise,3,4)
weather$hSunrise=as.numeric(weather$hSunrise);weather$mSunrise=as.numeric(weather$mSunrise)
# Sunset
weather$Sunset=as.character(weather$Sunset)
for(i in which(weather$Sunset=="-")){
  weather$Sunset[i]=weather$Sunset[i-1]}
weather$hSunset=substr(weather$Sunset,1,2)
weather$mSunset=substr(weather$Sunset,3,4)
weather$hSunset=as.numeric(weather$hSunset);weather$mSunset=as.numeric(weather$mSunset)
#Depth
weather$Depth=as.character(weather$Depth)
weather$Depth[which(weather$Depth=="M")]=0
weather$Depth=as.numeric(weather$Depth)
# SnowFall
weather$SnowFall=as.character(weather$SnowFall)
weather$SnowFall[which(weather$SnowFall=="  T")]=0.01
for(i in which(weather$SnowFall=="M")){
  if(i%%2==0){weather$SnowFall[i]=weather$SnowFall[i-1]}
  else{weather$SnowFall[i]=weather$SnowFall[i+1]}}
weather$SnowFall=as.numeric(weather$SnowFall)
# PrecipTotal
weather$PrecipTotal=as.character(weather$PrecipTotal)
weather$PrecipTotal[which(weather$PrecipTotal=="  T")]=0.001
for(i in which(weather$PrecipTotal=="M")){
  if(i%%2==0){weather$PrecipTotal[i]=weather$PrecipTotal[i-1]}
  else{weather$PrecipTotal[i]=weather$PrecipTotal[i+1]}}
weather$PrecipTotal=as.numeric(weather$PrecipTotal)
# StnPressure
weather$StnPressure=as.character(weather$StnPressure)
for(i in which(weather$StnPressure=="M")){
  if(i%%2==0){weather$StnPressure[i]=weather$StnPressure[i-1]}
  else{weather$StnPressure[i]=weather$StnPressure[i+1]}}
weather$StnPressure[c(2411,2412)]=0
weather$StnPressure=as.numeric(weather$StnPressure)
weather$StnPressure[2411]=mean(weather$StnPressure[-c(2411,2412)])
weather$StnPressure[2412]=weather$StnPressure[2411]
# SeaLevel
weather$SeaLevel=as.character(weather$SeaLevel)
for(i in which(weather$SeaLevel=="M")){
  if(i%%2==0){weather$SeaLevel[i]=weather$SeaLevel[i-1]}
  else{weather$SeaLevel[i]=weather$SeaLevel[i+1]}}
weather$SeaLevel=as.numeric(weather$SeaLevel)
# AvgSpeed
weather$AvgSpeed=as.character(weather$AvgSpeed)
for(i in which(weather$AvgSpeed=="M")){
  weather$AvgSpeed[i]=weather$AvgSpeed[i-1]}
weather$AvgSpeed=as.numeric(weather$AvgSpeed)
# Remove useless feature
weather=weather[,-c(11:15)]

# Calculate the distence between two station, merge "weather" ==============
library(geosphere)
distance <- function(longitude, latitude) {
  #Euclidian distances aren't accurate because we are on a sphere
  dist1 <- sqrt((stations[1,]$Latitude-latitude)^2+(stations[1,]$Longitude-longitude)^2)
  dist2 <- sqrt((stations[2,]$Latitude-latitude)^2+(stations[2,]$Longitude-longitude)^2)
  #Instead, let's use distHaversine from geosphere
  #Haversine distance : http://en.wikipedia.org/wiki/Haversine_formula 
  #dist1 <- distHaversine(c(stations[1,]$Longitude,stations[1,]$Latitude),c(longitude,latitude))
  #dist2 <- distHaversine(c(stations[2,]$Longitude,stations[2,]$Latitude),c(longitude,latitude)) 
  if(dist1<dist2){
    return(1)}
  return(2)}
stations<-data.frame(c(1,2),c(41.995,41.786),c(-87.933,-87.752))
names(stations)<-c("Station","Latitude","Longitude")
train$Station<-mapply(distance,train$Longitude,train$Latitude)
test$Station<-mapply(distance,test$Longitude,test$Latitude)
train=merge(train,weather, by=c("Date","Station"))
test=merge(test,weather, by=c("Date","Station"))
rm(weather);rm(stations)
test=test[order(test$Id),]

# Saperate Date and trap ===================================================
# trap
wx<-unique(train$Trap);coro<-c()
for (i in 1:length(wx)){  coro=rbind(coro,train[which(train$Trap==wx[i])[1],9:10])}
temp<-setdiff(test$Trap,train$Trap)
for (j in 1:length(temp)){
  ee=apply(coro,1,function(x) dist(rbind(x,test[which(test$Trap==temp[j])[1],10:11])))
  test[which(test$Trap==temp[j]),'Trap']=wx[which(ee==min(ee))[1]]}
test$Trap=factor(test$Trap)
rm(coro);rm(ee);rm(temp);rm(wx)

# time
train$dMonth=substr(train$Date,6,7)
train$dYear=substr(train$Date,1,4)
train$Date = as.Date(train$Date, format="%Y-%m-%d")
trainsDate = as.Date(paste0(train$dYear, "0101"), format="%Y%m%d")
train$dWeek = as.numeric(paste(floor((train$Date - trainsDate + 1)/7)))
test$dMonth=substr(test$Date,6,7)
test$dYear=substr(test$Date,1,4)
test$Date=as.Date(test$Date, format="%Y-%m-%d")
testsDate = as.Date(paste0(test$dYear, "0101"), format="%Y%m%d")
test$dWeek = as.numeric(paste(floor((test$Date - testsDate + 1)/7)))
spray$dMonth=substr(spray$Date,6,7)
spray$dYear=substr(spray$Date,1,4)
spray$Date=as.Date(spray$Date, format="%Y-%m-%d")
spraysDate = as.Date(paste0(spray$dYear, "0101"), format="%Y%m%d")
spray$dWeek = as.numeric(paste(floor((spray$Date - spraysDate + 1)/7)))

# Dealing with missing Species =============================================
# missing factor(Species)
vSpecies<-c(as.character(train$Species),as.character(test$Species))
vSpecies[vSpecies=="UNSPECIFIED CULEX"]<-"CULEX RESTUANS"
vSpecies<-factor(vSpecies,levels=unique(vSpecies))
train$Species=factor(vSpecies[1:nrow(train)],levels=unique(vSpecies))
test$Species=factor(vSpecies[(nrow(train)+1):length(vSpecies)],levels=unique(vSpecies))
rm(vSpecies)

# merge spray ==============================================================
train$spray <- rep(0,dim(train)[1])
for(i in 1:dim(train)[1]){
  day_diff <- as.Date(spray$Date, format="%Y-%m-%d")-(as.Date(train$Date[i], format="%Y-%m-%d")-366)
  day_index <- day_diff < c(-7) & day_diff > c(-32)
  if(sum(day_index)>0){
    dis_result <- rdist( train[i,c(9,10)], spray[day_index,c(3,4)]) < 0.1    
    if(sum(dis_result & day_index[day_index]) >0 ) {
      train$spray[i] =1}}}
test$spray <- rep(0,dim(test)[1])
for(i in 1:dim(test)[1]){
  day_diff <- as.Date(spray$Date, format="%Y-%m-%d")-(as.Date(test$Date[i], format="%Y-%m-%d")-366)
  day_index <- day_diff < c(-7) & day_diff > c(-32)
  if(sum(day_index)>0){
    dis_result <- rdist( test[i,c(10,11)], spray[day_index,c(3,4)]) < 0.1    
    if(sum(dis_result & day_index[day_index]) >0 ) {
      test$spray[i] =1}}}
rm(spray); rm(dis_result)

# Training and testing =====================================================
trai=train[,-c(1:3,5:6,8,11)]
tes=test[,-c(1,2,4,6:7,9,12)]

write.csv(trai,"train_spray.csv",row.names=FALSE,quote=FALSE)
write.csv(tes,"test_spray.csv",row.names=FALSE,quote=FALSE)