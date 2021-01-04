#--------------------------- PRE-PROCESSING THE DATA ---------------------------

# Installing the packages required for pre-processing.

install.packages("dplyr")
install.packages("magrittr")
install.packages("knitr")
install.packages("reshape")
install.packages("caret")
install.packages("e1071")
install.packages("kernlab")
install.packages("neuralnet")
install.packages("class")
install.packages("gmodels")


library(gmodels)
library(dplyr)
library(magrittr)
library(knitr)
library(reshape)
library(caret)
library(e1071)
library(kernlab)
library(neuralnet)
library(class)


# Initialising the directory and using the data for pre-process

setwd("C:/Users/Hassan/Desktop/Machine Learning ICA work/KNNICA-Model/KNNICA")
BreastCancer <-
  read.csv(file = "BreastCancerOriginal.data",
           stringsAsFactors = FALSE,
           header = TRUE)

# Changing row names. Currently set from X1 to X5.

BreastCancer <-
  rename(
    BreastCancer,
    c(
      X1000025 = "Patient_ID",
      X5 = "Clump_Thickness",
      X1 = "Uniformity_of_Cell_Size",
      X1.1 = "Uniformity_of_Cell_Shape",
      X1.2 = "Marginal_Adhesion",
      X2 = "Single_Epithelial_Cell_Size",
      X1.3 = "Bare_Nuclei",
      X3 = "Bland_Chromatin",
      X1.4 = "Normal_Nucleoli",
      X1.5 = "Mitoses",
      X2.1 = "Class"
    )
  )

# Checking the data summary which includes the Mean, Mode and Median.
summary(BreastCancer)


# Deleting the rows that are not required due to we identified similar data to those.
BreastCancer[-c(139, 145, 158, 249, 275, 294, 321, 411, 617),] %>% head()
BreastCancer <-
  BreastCancer[-c(139, 145, 158, 249, 275, 294, 321, 411, 617), ]

# Replacing the missing data by its mode for the Bare Nuclei variable.
# If the class is 4 the data will be replaced with 10.
# If the class is 4 the data will be replaced with 1.

class(BreastCancer$Bare_Nuclei)
BreastCancer$Bare_Nuclei[BreastCancer$Bare_Nuclei == "?" &
                           BreastCancer$Class == "4"] <- "10"
BreastCancer$Bare_Nuclei[BreastCancer$Bare_Nuclei == "?" &
                           BreastCancer$Class == "2"] <- "1"

BreastCancer$Bare_Nuclei <- as.numeric(BreastCancer$Bare_Nuclei)
# BreastCancer$Bare_Nuclei <- NULL


#USE EITHER THIS LINE TO REPLACE
#BreastCancer$Class[BreastCancer$Class == "4"] <- "M"
#BreastCancer$Class[BreastCancer$Class == "2"] <- "B"

# OR USE THIS LINE TO REPLACE
BreastCancer$Class <-
  factor(
    BreastCancer$Class,
    levels = c("2", "4"),
    labels = c("Benign", "Malignant")
  )

# BreastCancer$Class <- as.character.numeric_version(BreastCancer$Class)
# BreastCancer$Bare_Nuclei <- as.integer(BreastCancer$Bare_Nuclei)


table(BreastCancer$Class)

round(prop.table(table(BreastCancer$Class)) * 100, digits = 1)



summary(BreastCancer[c(
  "Clump_Thickness",
  "Uniformity_of_Cell_Size",
  "Uniformity_of_Cell_Shape",
  "Marginal_Adhesion",
  "Single_Epithelial_Cell_Size",
  "Bland_Chromatin",
  "Normal_Nucleoli",
  "Mitoses",
  "Class"
)])




normalize <- function (x) {return ((x - min(x)) / (max(x) - min(x)))}

normalize(c(1, 2, 3, 4, 5))
normalize(c(10, 20, 30, 40, 50))


Normalised_BreastCancer <- as.data.frame(lapply(BreastCancer[, 2:10], normalize))

head(Normalised_BreastCancer)

set.seed(123)

BreastCancer3 <- sample(1:nrow(Normalised_BreastCancer),
    size = nrow(Normalised_BreastCancer) * 0.7,
    replace = FALSE)

train_BC <- Normalised_BreastCancer[BreastCancer3, ]
test_BC <- Normalised_BreastCancer[-BreastCancer3, ]


train.BC_labels <- BreastCancer[BreastCancer3, 11]
test.BC_labels <- BreastCancer[-BreastCancer3, 11]


NROW(train.BC_labels)


BC_Test_Prediction <- knn(train = train_BC, test = test_BC, cl = train.BC_labels, k = 21)
BC_Test_Prediction2 <- knn(train = train_BC, test = test_BC, cl = train.BC_labels, k = 22)

BC_Test_Prediction_Accuracy <- 100 * sum(test.BC_labels == BC_Test_Prediction) / NROW(test.BC_labels)
BC_Test_Prediction_Accuracy2 <- 100 * sum(test.BC_labels == BC_Test_Prediction2) / NROW(test.BC_labels)


BC_Test_Prediction_Accuracy
BC_Test_Prediction_Accuracy2

table(BC_Test_Prediction , test.BC_labels)

table(BC_Test_Prediction2 , test.BC_labels)

confusionMatrix(table(BC_Test_Prediction , test.BC_labels))

i = 1
k.optm = 1

for (i in 1:50) {
  knn.mod <- knn(train = train_BC, test = test_BC, cl = train.BC_labels,k = i)
  
k.optm[i] <- 100 * sum(test.BC_labels == knn.mod) / NROW(test.BC_labels)
k = i
cat(k, '=', k.optm[i], '')
}