# crypto_pred

This repo includes training and STDIN/STDOUT deployment files for the given assignment.

The train_crypto.py includes the scrapping part using beautifulsoup and some pre processing and training a LSTM model part. LSTM model is choosen in oreder to exploit the character level dependencies in the wallet adress. Length of wallet adress is taken into account and concateneted with the output of the LSTM and then put through MLP layer.

Crypto.py file is the deployment file.

The runnig script for the crypto.py file is provided by the name crp.sh. in thsi it will read the STDIN input which should be given as a list of address as shown in .sh file and it will output the crypto currencies on the terminal as well as a csv file.

for training the argument of saving the model as --model output should be given while for the deployment the arguments of saved model --model_path and the directory for the final output csv --output_dir should be given.
