options(xtable.comment = FALSE)
carga_paquetes <- function() {
  dependencies = c("plyr",
                   "dplyr",
                   "knitr",
                   "ggplot2",
                   "dplyr",
                   "tidyr",
                   "apsrtable",
                   "xtable",
                   "tidyr",
                   "stringr",
                   "lubridate",
                   "foreign",
                   "readr",
                   "xtable",
                   "rmarkdown",
                   "devtools",
                   'rvest',
                   'doMC',
                   "stargazer",
                   "Hmisc")
  installed_packages = rownames(installed.packages())
  lapply(dependencies, function(x){
    if (!(x %in% installed_packages)){
      install.packages(x, repos="http://cran.us.r-project.org")
    }
    library(x, character.only=TRUE)
  })
}

imprime_tabla <- function(df, titulo, lab,archivo='', digits=0,...){  
  x <- xtable(df, digits = digits,
              caption = titulo,
              label = lab)
  print(x, table.placement='H', file=archivo,
        caption.placement ="top",
        include.rownames = F  )}


imprime_modelos <- function(lista_mod,titulo, lab, archivo){
  stargazer(lista_mod,title = titulo, type = 'latex',
            intercept.bottom = FALSE,
            label = lab,
            out = 'aux.tex',
                 table.placement = 'H',omit.stat="f",float = T) 
  system( paste('cat aux.tex', " | sh limpia_tabla_modelo.sh >",archivo ))
  system('rm aux.tex')
}

require(ggplot2)

tema_mapas <- theme(axis.line=element_blank(),
                    axis.text.x=element_blank(),
                    axis.text.y=element_blank(),
                    axis.ticks=element_blank(),
                    axis.title.x=element_blank(),
                    axis.title.y=element_blank(),
                    panel.background=element_blank(),
                    panel.border=element_blank(),
                    panel.grid.major=element_blank(),
                    panel.grid.minor=element_blank(),
                    plot.background=element_blank(),
                    legend.position="bottom")

eda_1 <- function(datos,var_cat_1,var_cat_2=NULL, var_cat_3=NULL){
  tipos <- sapply(names(datos),function(i){class(datos[,i])})
  tipos_num <- names(which(tipos=='numeric'))
    #   var_cat_1='V1'
  #   var_cat_2='V2'
  #   var_cat_3='V3'
  a_ply(tipos_num  ,  .margins = 1, .fun = function(var){
    print(var)
    
    p <-   ggplot(datos, aes_string(x=var_cat_1, y=var , colour=var_cat_2) ) + 
      geom_boxplot() + 
      geom_violin() + 
      geom_jitter()
    if(!is.null(var_cat_3)){p <- p+ facet_wrap(var_cat_3 ) }
    print(p)
    
  }) 
}


matrix_cor_nas <- function(df){
  x <- as.data.frame(abs(is.na(df))) 
  y <- x[which(sapply(x, sd) > 0)] 
  # Da la correación un valor alto positivo significa que desaparecen juntas.
  cor(y) 
}


indicesConNAs <- function(data, porcentaje=0.2) {
  n <- if (porcentaje < 1) {
    as.integer(porcentaje  * ncol(data))
  } else {
    stop("Debes de introducir el porcentaje de columnas con NAs.")
  }
  indices <- which( apply(data, 1, function(x) sum(is.na(x))) > n )
  if (!length(indices)) {
    warning("No hay observaciones con tantos NAs 
            (la respuesta de la función es vacía),
            no se recomienda indexar el data.frame con esto")
  }
  indices
  }



# Función:
normalizarNombres <- function(df.colnames) {
  require(stringr)
  df.colnames <- str_replace_all(df.colnames,"([[:punct:]])|\\s+",".")
  df.colnames <- gsub("([a-z])([A-Z])", "\\1.\\L\\2", df.colnames, 
                      perl = TRUE)
  df.colnames  <- sub("^(.[a-z])", "\\L\\1", df.colnames, perl = TRUE)
  df.colnames  <- tolower(df.colnames)
}


pruebaChiCuadrada <- function(var1, var2) {
  tabla <- table(var1, var2)
  
  plot(tabla, main=var1, las=1)
  
  print(var1)
  
  print(chisq.test(tabla))
}


## Predicciones


predecirVariableCategorica <- function(outCol, varCol, appCol, pos) {
  pPos <- sum(outCol == pos) / length(outCol)
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[pos]
  vTab <- table(as.factor(outCol), varCol)
  pPosWv <- (vTab[pos,]+1.0e-3*pPos)/(colistaums(vTab)+1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(appCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred
}


## Evaluación

library(ROCR)

calcAUC <- function(predcol, outcol) {
  perf <- performance(prediction(predcol, outcol == pos), 'auc')
  as.numeric(perf@y.values)
}



#### FEATURE SELECTION

#### low variability

low.variability  <- function(df){
  options("scipen"=100, "digits"=4) # indico quitar expr cientifica
  quantile  <- apply(df, 2, FUN = quantile) # visualizo los quantiles
  IQR  <- apply(df,2,FUN=IQR) # visualizdo el IQR
  var  <- (apply(df,2,FUN=IQR) /
             (apply(df,2,FUN=max)-apply(df,2,FUN=min)))*100 #IQR/rango
  tb  <- round(rbind(quantile,IQR,var),4)
  tb 
}



### Feature selecction 

correlation.filtering  <- function(df){
  require(corrgram)
  cor  <-  cor(df,use="complete.obs")
  print(corrgram(df, order=TRUE, lower.panel=panel.shade,
                 upper.panel=panel.pie, text.panel=panel.txt,
                 main="Feature Selection") )
  cor 
}

#### Fast correlation-based filtering 
FCB.filtering  <- function(df,y){
  # df dataframe
  # y la variable objetivo
  df  <- as.data.frame(cor(sub_num,use="complete.obs"))
  df2  <- as.data.frame(df[with(df,order(-y)),])
  df2  <- df2[df2==1]  <-  0
  muy_cor <- which.max(df2[2,] > .85)
  na  <- names(df2)[muy_cor]
  df2 <- df2[-muy_cor]
  df2 
}

epsilon <- function( datos , y , tol=1){
  
  require(plyr)
  require(dplyr)
  datos.c <- datos[ which( !sapply( datos, is.numeric))]
  vars.cat <- names( datos.c )[names( datos.c)!=y]
  lista <- c()
  n <- length(vars.cat)
  
  for(i in 1:n){
    
    group1 <- datos.c %>% 
      group_by(feature=datos.c[,vars.cat[i]]) %>% 
      summarise(proba.Temporada=n()/dim(datos)[1])
    
    group2 <- datos.c %>% 
      group_by( Clase = datos.c[,y]) %>%
      dplyr::summarise(PC=n()/dim(datos)[1])
    
    tablaEpsilon<- datos.c %>% 
      group_by(Clase=datos.c[,y] ,
               feature=datos.c[, vars.cat[i] ] ) %>%
      dplyr::summarise( Nx=n(),
                        prob=n()/nrow(datos) ) %>% 
      merge( group1, by="feature" ) %>%
      merge( group2, by="Clase" ) %>%
      dplyr::mutate( epsilon = Nx*( prob/ proba.Temporada - PC)/sqrt(Nx * PC* (1-PC) ) )
    
    eps <- max(tablaEpsilon$epsilon)
    
    if(eps<tol){
      lista<- c( lista, vars.cat[i] )
    }
  }
  
  vars.num<- colnames( datos[ which( sapply( datos, is.numeric))])
  datos.num <-datos[, c( y, vars.num)]
  
  datos.num[2: ncol(datos.num) ] <- scale(datos.num[ 2:ncol(datos.num)],
                                          center=TRUE,scale=TRUE)
  
  n <- length(vars.num)
  
  clases <- unique(datos.num[,y])
  m <- length(clases)
  
  for(i in 1:n){  
    for(j in 1:m){
      df1 <- datos.num[ datos.num[,y] == as.character( clases[j] ) , ]
      N1<-nrow(df1)
      mean1<-mean(df1[,vars.num[i]],na.rm=TRUE)
      var1<-var(df1[,vars.num[i]],na.rm=TRUE)
      
      df2 <- datos.num[ datos.num[,y] != as.character( clases[j] ), ]
      N2 <- nrow(df2)
      
      mean2 <- mean(df2[,vars.num[i]],na.rm=TRUE)
      var2 <- var(df2[,vars.num[i]],na.rm=TRUE)
      eps.num <- -1e6
      x <- ifelse(is.nan((mean1-mean2) / sqrt( var1/N1 - var2/N2 ))
                  , eps.num,
                  (mean1-mean2) / sqrt( var1/N1 - var2/N2 ))
      eps.num <- ifelse( eps.num <  x,
                         eps.num <- x,
                         eps.num )
    }
    if( eps.num < tol ){
      lista<-c( lista , vars.num[i])
    }
  }
  return(lista)
}



# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}



genera_tabla_larga <- function(datos,
                               titulo_tabla,
                               label_tabla,
                               archivo_salida='tabla.tex', 
                               digits_tabla=0){
  
  align_tabla <- paste0(rep('X',ncol(datos)+1),collapse = '')
  
  tabla <- xtable( datos  , 
                   caption = titulo_tabla,
                   label = label_tabla, 
                   align = align_tabla , 
                   digits = 0)
  if(nrow(datos)>20){
    #Para cuando hay tablas muy largas
    add.to.row <- list(pos = list(0), command = NULL)
    
    command <- paste0("\\hline\n\\endhead\n",
                      "\\hline\n\\multicolumn{", dim(datos)[2],"}{l}", "{\\tiny Continues on next page}", "\\endfoot\n",
                      "\\multicolumn{", dim(datos)[2],"}{l}","{\\tiny End of table}",
                      "\\endlastfoot\n")
    add.to.row$command <- command
    
    x <- print( tabla, 
                print.results = getOption("xtable.print.results", F),
                floating=F,
                include.rownames = F, 
                include.colnames = T,
                latex.environments = getOption("xtable.latex.environments", c("center")),
                tabular.environment = "longtable",
                caption.placement = 'top',
                add.to.row = add.to.row
                # hline.after = c(-1) 
    )  
    cat(paste0('% !TEX root = ../latex_detecting_corruption_collusion_fraud/A0_thesis_detecting_collusion_corruption_fraud.tex\n\\begin{filecontents}{longtab.tex}\n\\scriptsize\n',x , '\n\\end{filecontents}\n\\LTXtable{1.0\\linewidth}{longtab.tex}'), 
        file =archivo_salida )
    cat(paste0('Se generó el archivo de salida: \"',archivo_salida,
               '\".\nSe requieren los paquetes:\n\\usepackage{longtable}\n\\usepackage{tabularx}\n\\usepackage{filecontents,ltxtable}\n\nEn latex sólo pones \\input{',archivo_salida,'}'))
  }else{
    x <- print( tabla, 
                print.results = getOption("xtable.print.results", F),
                floating=F,
                include.rownames = F, 
                include.colnames = T,
                latex.environments = getOption("xtable.latex.environments", c("center")),
                tabular.environment = "tabularx",
                caption.placement = 'top',
                hline.after = c(-1,0)
    )
    cat(paste0('% !TEX root = ../latex_detecting_corruption_collusion_fraud/A0_thesis_detecting_collusion_corruption_fraud.tex\n',x), file = archivo_salida)
    cat(paste0('Se generó el archivo de salida: \"',archivo_salida,
               '\".\nSe requieren los paquetes:\n\\usepackage{longtable}\n\\usepackage{tabularx}\n\\usepackage{filecontents,ltxtable}\n\nEn latex sólo pones \\input{',archivo_salida,'}'))
  }
}

