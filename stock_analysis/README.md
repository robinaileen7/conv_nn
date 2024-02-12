I'm doing this personal project to take a chance to learn Convolutional Neural Network (CNN) in PyTorch. Thanks for all the amazing tutorials online; those are more helpful than I thought! 
I collected largest food and tech companies' stock data from Yahoo Finance and compute their time-serial returns on different frequencies (e.g. from daily return all the way to yearly return). Next, I used matplotlib's imshow to read data as images (I know there are better ways to feed in those data but I did it for an exercise). Starting from there, I use torch.nn to build a neural network to learn about those images' features, and then apply linear transformation to the final output into the size of classification I want.
On top of those, for each batch of my training data, I use Adam algorithm to update parameters on each step, and then use the best parameters to compute test sample accruracy.
As for now, I have tried classification on stock sectors and stock performaces. There are a lot more for me to keep polishing such as how to convert the stock returns into something CNN can better initialize, and how to achieve higher accruacy by improving optimization methods.