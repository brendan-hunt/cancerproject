library(caret)
library(glmnet)
library(pROC)


df <- read.csv("data.csv")
df$X <- NULL
df$id <- NULL
df$diagnosis <- as.factor(df$diagnosis)

stopifnot(sum(is.na(df)) == 0)
class(df$diagnosis)   
levels(df$diagnosis) 

#80/20 split
set.seed(42)
train_idx <- createDataPartition(df$diagnosis, p = 0.8, list = FALSE)
train_data <- df[train_idx, ]
test_data  <- df[-train_idx, ]

x_train <- as.matrix(train_data[, -1])
y_train <- train_data$diagnosis
x_test  <- as.matrix(test_data[, -1])
y_test  <- test_data$diagnosis

class(y_test)   
levels(y_test)  

cv_fit <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0, standardize = TRUE)
best_lambda <- cv_fit$lambda.min

probs <- predict(cv_fit, newx = x_test, s = best_lambda, type = "response")[, 1]
preds <- ifelse(probs > 0.5, "M", "B")
preds <- factor(preds, levels = levels(y_test))

cm <- confusionMatrix(preds, y_test, positive = "M")
print(cm)

colnames(x_train)

roc_obj <- roc(y_test, probs, levels = c("B", "M"))
cat("AUC:", auc(roc_obj), "\n")
plot(roc_obj, main = "ROC Curve - Ridge Logistic Regression")


vif_check_data <- train_data[, c("diagnosis","radius_mean","texture_mean",
                                  "smoothness_mean","compactness_mean",
                                  "symmetry_mean","fractal_dimension_mean")]
reduced_model <- glm(diagnosis ~ ., data = vif_check_data, family = binomial)
summary(reduced_model)

full_model <- glm(diagnosis ~ ., data = train_data, family = binomial)
summary(full_model)

install.packages("reshape2") 
library(reshape2)

corr_matrix <- cor(train_data[, -1]) 
corr_melt <- melt(corr_matrix)

ggplot(corr_melt, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 7),
        axis.text.y = element_text(size = 7)) +
  labs(title = "Feature Correlation Heatmap", x = "", y = "", fill = "Correlation")