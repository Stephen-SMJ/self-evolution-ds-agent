# This example is for train

rm(list=ls())
gc()

setwd('..')

library(jsonlite)
library(doParallel)
library(foreach)

train_json <- fromJSON('train.json')

json_df <- function(data_json){
  df_1 <- as.data.frame(data_json$images)
  df_2 <- as.data.frame(data_json$annotations)
  df <- merge(df_1, df_2, by = 'imageId', all = TRUE)
  return(df)
}


train_df <- json_df(train_json)

load_save_image <- function(data_df, dir_save){
  error <- foreach(i = 1:nrow(data_df), .combine = 'c',
                   .verbose=TRUE)%dopar%{
    setwd(dir_save)
    indice <- data_df$imageId[i]
    url <- data_df$url[i][[1]]
    save_name <- paste(indice, '.jpg', sep='')
    err <- c()
    tryCatch(download.file(url, save_name,mode = 'wb'),
             error = function(e){ 
               print(e)
               print(indice)
               return(indice)
               err <<- c(err, indice)
               })
  }
  print('all loaded')
  return(error)
}

library(parallel)
library(doParallel)
registerDoParallel(cores = 8)

load_save_image(data_df = train_df,
                dir_save = 'F:/CONCURSOS KAGGLE/01_identifica_ropa/03_imagenes/01_train/')
