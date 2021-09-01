from flask import Flask, render_template, request
import datetime
from Transformermodel import TransAm
from Postional import PositionalEncoding
from prediction_fun import predicition
app=Flask(__name__,template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
#delete unnecessary methods
#def home_page():

#return render_template('index.html', title='Home')'''
def getPrediction():
    #date=""
    price_1h=0
    price_1d=0
    price_1w=0


    if  request.method=='POST':
        date=request.form['date']
        if request.form['button']=='Predict':
            print(date)

            #call the function
            #prices=Yuchong's_function()

            #get price 1h ahead
            price_1h=5

            #get prices 1d ahead
            price_1d=8

            #get prices 1w ahead
            price_1w=11

    return render_template('index.html', price_1h=price_1h, price_1d=price_1d, price_1w=price_1w)


@app.route('/getPrediction/')
def getPrediction_By_Date():
    # initial the value
    price_1h, price_1d, price_1w = 0,0,0
    date = datetime.datetime.now()

    if(request.args.get('date')):

        # get the chosen date
        date = request.args.get('date')
        print(date)
        my_date = datetime.datetime.strptime(date, "%d/%m/%Y - %H:%M")
        #my_date = datetime.strptime(date, "%Y/%m/%d-%H:00")
        date_=my_date.date().strftime('%Y-%m-%d')
        print('date',date)
        time=my_date.time().hour
        #int(time)
        print(time)

        # get price 1h ahead
        combine =predicition(load_trans_model = "96-48.pkl",
               load_lstm_model = "epoch=12-step=649.ckpt",
               choose_date = date_,
               choose_hour = time,
               devices = "cpu")

        price_1h = combine[0]

        # get prices 1d ahead
        price_1d = combine[23]
        # get prices 1w ahead
        price_1w = combine[-1]

    print(date)
    return render_template('index.html', date=date, price_1h=price_1h, price_1d=price_1d, price_1w=price_1w)

if __name__=='__main__':
    app.run(port=8309)