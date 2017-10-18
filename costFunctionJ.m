{\rtf1\ansi\ansicpg936\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww14880\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs28 \cf0 function J = costFunctionJ(X, y, theta)\
\
%X is the \'93design matrix\'94 containing our training examples.\
%y is the class labels\
\
m = size(X, 1);				%number of training examples\
predictions = X*theta			%predictions of hypothesis on all m examples\
sqrErrors = (predictions - y) .^ 2		%squared errors\
\
J = 1/(2*m) * sum(sqrErrors);}