import pandas as pd
import os


def logger(cv_fold, epoch, lr, training_loss, val_loss, val_kappa, filename='logger.csv'):
	if not os.path.exists(filename):
		with open(filename, 'a') as f:
			f.write('epoch, cv_fold, lr, training_loss, val_loss, val_kappa\n')
	
	with open(filename, 'a') as f:
		f.write(str(epoch)+', '+str(cv_fold)+', '+str(lr)+', '+str(training_loss)+', '+str(val_loss)+', '+str(val_kappa)+'\n')

