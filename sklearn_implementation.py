import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ConfusionMatrix

df=pd.read_excel('C:\!Devasheesh resources\CS235_DM\DryBeanDataset\DryBeanDataset\Dry_Bean_Dataset.xlsx')

#Using Pearson Correlation for correlation matrix
plt.figure(figsize=(19,19))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
sns.pairplot(df)
plt.show()


x=df.drop(['Class','Perimeter','MajorAxisLength','MinorAxisLength','ConvexArea'], axis=1)
y=df['Class']


#Normalizing the data: Z score ('StandardScaler' is used)
scaler = StandardScaler()
scaler.fit(x)
scaled = scaler.transform(x)
x = pd.DataFrame(scaled, columns=x.columns)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=9)

dtree=DecisionTreeClassifier(max_depth=9,max_features='sqrt')
dtree.fit(x_train,y_train)
prediction=dtree.predict(x_test)

#getting the traditional details
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

# Using yellowbrick to plot the conf matrix
opt_classes=['BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA']
bean_confmat = ConfusionMatrix(dtree, classes=opt_classes, color='bold')
bean_confmat.score(x_test, y_test)
bean_confmat.show()
