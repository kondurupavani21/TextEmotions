import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
df = pd.read_csv("FINALDATA.csv")
#specifing features and target
X=df['sample']
y=df['emotion']
#split data
model=MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
vector=CountVectorizer()
X_train_vectors=vector.fit_transform(X_train)
X_test_vectors=vector.transform(X_test)
#train the model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")
label={
    0:'',#anger
    1:'',#fear
    2:'',#joy
    3:'',#love
    4:'',#sadness
    5:''#surprise
}
user_input=input("Tell me What's happening:")
custom_message = [user_input]
custom_vector = vector.transform(custom_message)
prediction = model.predict(custom_vector)
print(prediction)
p=prediction[0]
print(f"Predicted emotion: {label[p]}")