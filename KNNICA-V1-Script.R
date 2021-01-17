#--------------------------- PRE-PROCESSING THE DATA ---------------------------



# Installing the packages required for pre-processing and to work with the model later.
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
install.packages("Metrics")


# Initialising the libraries.
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
library('Metrics')


# Initialising the directory and using the data for pre-process
# WORKING WITH THIS DATA FROM HOME
setwd("C:/Users/Hassan/Desktop/Machine Learning ICA work/KNNICA-Model/KNNICA")

#setwd("C/:Users/V8039087/Desktop/MACHINE LEARNING FULL ICA WORK/KNNICA-master/KNNICA-master")

# WORKING WITH THIS DATA FROM UNI
#setwd("U:/YEAR 3 COURSE WORK/ML ICA")



# Reading the file which contains the data and loading it to the workspace.
BreastCancer <-
  read.csv(file = "BreastCancerOriginal.data",
           stringsAsFactors = FALSE,
           header = TRUE)



# Changing variable names. Currently set from X1 to X5.
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

# Replacing to factor will help to get the mode.
BreastCancer$Bare_Nuclei <- as.factor(BreastCancer$Bare_Nuclei)

# Checking the summary for BareNuclei variable where the class is 2 and 4.
# This should give us the mode for both which later will be used to replace the data.
summary (BreastCancer$Bare_Nuclei[BreastCancer$Class == "4"])
summary (BreastCancer$Bare_Nuclei[BreastCancer$Class == "2"])



# Deleting the rows that are not required due to I identified similar data to those.
BreastCancer[-c(139, 145, 158, 249, 275, 294, 321, 411, 617), ] %>% head()
BreastCancer <-
  BreastCancer[-c(139, 145, 158, 249, 275, 294, 321, 411, 617),]


# Replacing the missing data by its mode for the Bare Nuclei variable.
# If the class is 4 the data will be replaced with 10.
class(BreastCancer$Bare_Nuclei)
BreastCancer$Bare_Nuclei[BreastCancer$Bare_Nuclei == "?" &
                           BreastCancer$Class == "4"] <- "10"


# If the class is 2 the data will be replaced with 1.
BreastCancer$Bare_Nuclei[BreastCancer$Bare_Nuclei == "?" &
                           BreastCancer$Class == "2"] <- "1"



# Converting the Bare Nuclei variable to numeric as currently it was set as Character.
BreastCancer$Bare_Nuclei <- as.numeric(BreastCancer$Bare_Nuclei)



# Deleting the Bare Nuclei Variable.
# BreastCancer$Bare_Nuclei <- NULL



# Replacing the class variable data so it is better understandable.
# BreastCancer$Class[BreastCancer$Class == "4"] <- "M"
# BreastCancer$Class[BreastCancer$Class == "2"] <- "B"
BreastCancer$Class <-
  factor(
    BreastCancer$Class,
    levels = c("2", "4"),
    labels = c("Benign", "Malignant")
  )




#----------------------------- WORKING WITH KNN -----------------------------



# Checking how many instances of each class there are (Benign and Malignant)
table(BreastCancer$Class)




# Checking the percentage of each class (Benign and Malignant)
round(prop.table(table(BreastCancer$Class)) * 100, digits = 1)



# Checking the summary again as there has been few  changes.
summary(BreastCancer[c(
  "Clump_Thickness",
  "Uniformity_of_Cell_Size",
  "Uniformity_of_Cell_Shape",
  "Marginal_Adhesion",
  "Single_Epithelial_Cell_Size",
  "Bare_Nuclei",
  "Bland_Chromatin",
  "Normal_Nucleoli",
  "Mitoses",
  "Class"
)])




# In this function I will try to normalize the numeric data.
# The normalize function takes a vector X of values that are numeric, and for each value in X, it will subtract the
# minimum value and then divide by the range of values in X.
# At the end, the resulting vector is returned.
normalize <- function (x) {
  return ((x - min(x)) / (max(x) - min(x)))
}



# Testing if the normalize method above is working.
# The data normalised in the second vector are larger than the first one
# After normalisation, they both become equal.
normalize(c(1, 2, 3, 4, 5))
normalize(c(10, 20, 30, 40, 50))



# I have stored the normalised data in the ‘BreastCancer_Normalisation’ data frame
# The ‘Patient_ID’ variable was not included because it is not required.
# I also excluded the ‘Class’ variable as it is the variable I am trying to predict the accuracy for.
BreastCancer_Normalisation <-
  as.data.frame(lapply(BreastCancer[2:10], normalize))
head(BreastCancer_Normalisation)



# The following set.seed() function will help to reuse the same set of random variables.
# It might be required further in the ICA to evaluate particular task again with same random varibales.
set.seed(295)


# The sample function takes the normalised data that I created above where the class variable was taken off and it distributes into 70/30 proportion.
BreastCancer3 <-
  sample(
    1:nrow(BreastCancer_Normalisation),
    size = nrow(BreastCancer_Normalisation) * 0.7,
    replace = FALSE
  )

# The BC_Train is using 70% of data specified above in the BreastCancer3 data we created.
BC_Train <- BreastCancer_Normalisation[BreastCancer3, ]
# The BC_Test is using the remaining data which should be 30% specified above in the BreastCancer3 data we created.
BC_Test <- BreastCancer_Normalisation[-BreastCancer3, ]


# When I create the training and test data, I excluded the target variable which is the CLASS variable.
# The labels I am creating below are stored in a separate factor vectors.
# 11 stands for the 11th variable which was the Class.
# The training labels are training for the Class variable.
BC_Train_Labels <- BreastCancer[BreastCancer3, 11]
# The testing labels are training for the Class variable.
BC_Test_Labels <- BreastCancer[-BreastCancer3, 11]


# The KNN function is used to classify the data and it returns a factor vector of predicated labels.
# I used 21 as it is the close to the square root of our training data.
# We are training the BC_Train data which was the 70% when I splitted.
# We are testing the BC_Test data which was the 30% when I splitted.
BC_Test_Prediction <-
  knn(train = BC_Train, test = BC_Test, cl = BC_Train_Labels, k = 21)

# Cross table is used to evaluate the model perfomance which is provided by GModels package.
# The prop.chisq = FALSE removes the chi-square values that are not needed.
# This gives us our TP. TN. FP. FN. Results
CrossTable(x = BC_Test_Labels, y = BC_Test_Prediction, prop.chisq = FALSE)


# The following line was used to get the overall accuracy for this model.
Metrics::accuracy(BC_Test_Labels, BC_Test_Prediction)


#------------ IMPROVING MODEL PERFOMANCE USING Z-Score Standardisation----------

# To improve the model we are following the same steps used before but this time using the Z-Score Standardisation.

# Create BreastCancer_Z which gets the data from my original BreastCancer DataSet but excludes the Class variable.
BreastCancer_Z <- as.data.frame(scale(BreastCancer[-11]))

# The following set.seed() function will help to reuse the same set of random variables.
# It might be required further in the ICA to evaluate particular task again with same random varibales.
set.seed(206)

# The sample function takes the normalised data that I created above where the class variable was taken off and it distributes into 70/30 proportion.
BreastCancerZ <- sample(1:nrow(BreastCancer_Z), size = nrow(BreastCancer_Z) * 0.7, replace = FALSE)

# The BC_TrainZ is using 70% of data specified above in the BreastCancerZ data we created.
BC_TrainZ <- BreastCancer_Z[BreastCancerZ, ]
# The BC_TestZ is using the remaining data which should be 30% specified above in the BreastCancerZ data we created.
BC_TestZ <- BreastCancer_Z[-BreastCancerZ, ]

# When I create the BreastCancer_Z data, I excluded the target variable which is the CLASS variable.
# The labels I am creating below are stored in a separate factor vectors.
# 11 stands for the 11th variable which was the Class from the original dataset.
# The training labels are training for the Class variable.
BC_Train_LabelsZ <- BreastCancer[BreastCancerZ, 11]
BC_Test_LabelsZ <- BreastCancer[-BreastCancerZ, 11]

# The KNN function is used to classify the data and it returns a factor vector of predicated labels.
# I used 21 as it is the close to the square root of our training data(483).
# We are training the BC_TrainZ data which was the 70% when I splitted.
# We are testing the BC_TestZ data which was the 30% when I splitted.
BC_Test_PredictionZ <-
  knn(train = BC_TrainZ, test = BC_TestZ, cl = BC_Train_LabelsZ, k = 21)

# Cross table is used to evaluate the model perfomance which is provided by GModels package.
# The prop.chisq = FALSE removes the chi-square values that are not needed.
# This gives us our TP. TN. FP. FN. Results
CrossTable(x = BC_Test_LabelsZ, y = BC_Test_PredictionZ, prop.chisq = FALSE)

# Dummy Plot
plot(x = BC_Test_LabelsZ, y = BC_Test_PredictionZ)



# ------------------------------------------------------------------------------
# Display the accuracies for both version that I have worked with.
Metrics::accuracy(BC_Test_Labels, BC_Test_Prediction)
Metrics::accuracy(BC_Test_LabelsZ, BC_Test_PredictionZ)
