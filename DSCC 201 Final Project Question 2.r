
wheat<- read.csv('/public/bmort/R/wheat.csv')

is.na(wheat)

missing_by_col <- colSums(is.na(wheat))

# extract column names with missing data
colnames_with_missing <- names(missing_by_col[missing_by_col > 0])

# print the column names with missing data
cat("Columns with missing data:", colnames_with_missing, "\n")


# calculate the mean of the yield column
width_mean <- mean(wheat$width, na.rm = TRUE)

# replace missing values in the yield column with the mean value
wheat$width <- ifelse(is.na(wheat$width), width_mean, wheat$width)

# print the number of missing values in the filled data frame
cat("Number of missing values in the filled data frame:", sum(is.na(wheat$width)), "\n")


summary(wheat)


# create a boxplot for each column
par(mfrow=c(3,3))
for (i in 1:7) {
  boxplot(wheat[,i], main = names(wheat)[i])
}


library(corrplot)

wheat_numeric <- wheat[, sapply(wheat, is.numeric)]
cor_matrix <- cor(wheat_numeric)
corrplot(cor_matrix, method="color", type="upper", 
         order="hclust", tl.col="black", tl.srt=45)


# Load the caret package and the wheat data set
library(caret)


# Set the seed for reproducibility
set.seed(123)

# Create an index of the rows to use for training
trainIndex <- createDataPartition(y = wheat$type, p = 0.8, list = FALSE)

# Create the training and testing data sets
trainData <- wheat[trainIndex, ]
testData <- wheat[-trainIndex, ]


trainData

testData


set.seed(123)

# Create a train control object for repeated cross-validation
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

# Train an SVM model with a linear kernel using all features
svmFit <- train(type ~ ., data = trainData, method = "svmLinear", trControl = ctrl)

# Print the model summary
print(svmFit)


predictions <- predict(svmFit, newdata = testData)

# Calculate the accuracy of the model on the test data set
accuracy <- mean(predictions == testData$type)
cat("Accuracy:", round(accuracy, 2))

wheat_unknown<- read.csv('/public/bmort/R/wheat-unknown.csv')

wheat_unknown

predicted_values <- predict(svmFit, newdata = wheat_unknown)
predicted_values<-data.frame(predicted_values)
predicted_values
