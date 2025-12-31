from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import os
from config import Config
from utils.database import init_db

from services.crop_ml_service import crop_ml_service
from services.disease_detection_service import disease_detection_service
from services.loan_ml_service import loan_ml_service

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['BASE_DIR'], 'data'), exist_ok=True)
os.makedirs(app.config['MODELS_DIR'], exist_ok=True)

init_db()


try:
    import pandas as pd
    
    market_prices_df = pd.read_csv('datasets/Crop_Data.csv')
    market_prices_df.columns = ['State', 'District', 'Market', 'Commodity', 
                                 'Min_Price', 'Max_Price', 'Modal_Price']
    
    crop_ml_df = pd.read_csv('datasets/Crop_recommendation.csv')
    production_df = pd.read_csv('datasets/crop_production.csv')  
    
    DATASETS_LOADED = True

except Exception as e:
    market_prices_df = pd.DataFrame()
    crop_ml_df = pd.DataFrame()
    production_df = pd.DataFrame()
    DATASETS_LOADED = False


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        soil_data = {
            'nitrogen': float(request.form.get('nitrogen', 50)),
            'phosphorus': float(request.form.get('phosphorus', 30)),
            'potassium': float(request.form.get('potassium', 180)),
            'ph': float(request.form.get('ph', 7.0))
        }
        
        min_temp = float(request.form.get('min_temp', 20))
        max_temp = float(request.form.get('max_temp', 30))
        avg_temp = (min_temp + max_temp) / 2
        
        climate_data = {
            'state': request.form.get('state'),
            'district': request.form.get('district'),
            'temperature': avg_temp,
            'humidity': float(request.form.get('humidity', 65)),
            'rainfall': float(request.form.get('rainfall', 680))
        }
        
        farm_data = {
            'farm_size': float(request.form.get('farm_size', 2.5)),
            'irrigation': request.form.get('irrigation_type', 'drip')
        }
        
        recommendations = crop_ml_service.recommend_crops(soil_data, climate_data, farm_data)
        
        return render_template('crop_results.html', recommendations=recommendations)
    
    states_list = []
    districts_by_state = {}
    
    if DATASETS_LOADED and not production_df.empty:
        try:
            states_list = sorted(production_df['State_Name'].unique().tolist())
            
            for state in states_list:
                state_df = production_df[production_df['State_Name'] == state]
                districts = sorted(state_df['District_Name'].unique().tolist())
                districts_by_state[state] = districts
        except Exception as e:
            print(f"Error loading states/districts: {e}")
    
    return render_template('crop_recommendation.html',states_list=states_list,districts_by_state=districts_by_state)


@app.route('/select-crop', methods=['POST'])
def select_crop():
    crop_name = request.form.get('crop_name')
    if crop_name:
        session['selected_crop'] = crop_name
        flash(f'Selected crop: {crop_name}. You can now use it in Loan Estimator or Storage Management.', 'success')
    return redirect(url_for('crop_recommendation'))


@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = disease_detection_service.predict(filepath)
            
            return render_template('disease_result.html', result=result)
        else:
            flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)', 'error')
            return redirect(request.url)
    
    return render_template('disease_detection.html')


@app.route('/loan-estimate', methods=['GET', 'POST'])
def loan_estimator():
    if request.method == 'POST':
        crop = request.form.get('crop', 'Wheat')
        farm_size = float(request.form.get('farm_size', 2.5))
        state = request.form.get('state', 'Maharashtra')
        interest_rate = float(request.form.get('interest_rate', 6.5))
        loan_duration = int(request.form.get('loan_duration', 12))
        
        result = loan_ml_service.estimate_loan(crop, state, farm_size)
        
        recommended_loan = result['recommended_loan']
        total_interest = recommended_loan * (interest_rate / 100)
        emi_monthly = (recommended_loan + total_interest) / loan_duration if loan_duration > 0 else 0
        
        result['interest_rate'] = interest_rate
        result['loan_duration'] = loan_duration
        result['emi_monthly'] = int(emi_monthly)
        result['total_interest'] = int(total_interest)
        result['total_investment'] = int(result.get('total_investment', 0))
        result['recommended_loan'] = int(result['recommended_loan'])
        result['gross_revenue'] = int(result['gross_revenue'])
        result['net_profit'] = int(result['net_profit'])
        result['roi'] = round(result['roi'], 2)
        
        return render_template('loan_result.html', result=result)
    
    import pandas as pd
    try:
        crop_rec_df = pd.read_csv('datasets/Crop_recommendation.csv')
        crop_list = sorted(crop_rec_df['label'].unique())
    except:
        crop_list = ['Rice', 'Wheat', 'Cotton', 'Maize']
    
    try:
        prod_df = pd.read_csv('datasets/crop_production.csv')
        states_list = sorted(prod_df['State_Name'].unique().tolist())
    except:
        states_list = ['Maharashtra', 'Punjab', 'Uttar Pradesh', 'Madhya Pradesh']
    
    return render_template('loan_estimator.html', available_crops=crop_list, states_list=states_list)


@app.route('/storage/register', methods=['GET', 'POST'])
def storage_register():
    from utils.database import insert_storage_batch, get_all_storage_batches
    
    if request.method == 'POST':
        batch_id = insert_storage_batch(
            user_id=1,
            crop_type=request.form.get('crop_type'),
            quantity=float(request.form.get('quantity')),
            harvest_date=request.form.get('harvest_date'),
            storage_date=request.form.get('storage_date'),
            temperature=float(request.form.get('temperature', 22)),
            humidity=float(request.form.get('humidity', 68)),
            storage_type=request.form.get('storage_type', 'Cooperative'),
            status='GREEN'
        )
        
        flash('Storage batch registered successfully!', 'success')
        return redirect(url_for('storage_register'))
    
    inventory = get_all_storage_batches()
    
    return render_template('storage_management.html', inventory=inventory)


@app.route('/storage/delete/<int:batch_id>', methods=['POST'])
def storage_delete(batch_id):
    from utils.database import delete_storage_batch
    
    if delete_storage_batch(batch_id):
        flash('Storage batch deleted successfully!', 'success')
    else:
        flash('Failed to delete batch. Please try again.', 'error')
    
    return redirect(url_for('storage_register'))


@app.route('/storage/dashboard/<int:batch_id>')
def storage_dashboard(batch_id):
    from utils.database import get_storage_batch
    
    batch = get_storage_batch(batch_id)
    
    if not batch:
        flash('Storage batch not found!', 'error')
        return redirect(url_for('storage_register'))
    
    from datetime import datetime, timedelta
    
    storage_date = datetime.strptime(batch['storage_date'], '%Y-%m-%d')
    days_stored = (datetime.now() - storage_date).days
    
    max_storage_days = {
        'Wheat': 365, 'Rice': 300, 'Chickpea': 300, 'Soybean': 240,
        'Cotton': 365, 'Corn': 180, 'Potato': 120, 'Onion': 240, 'Tomato': 30
    }
    
    max_days = max_storage_days.get(batch['crop_type'], 180)
    shelf_life_percent = min(100, (days_stored / max_days) * 100)
    
    if shelf_life_percent < 60:
        status = 'GREEN'
    elif shelf_life_percent < 80:
        status = 'YELLOW'
    elif shelf_life_percent < 90:
        status = 'ORANGE'
    else:
        status = 'RED'
    
    dashboard_data = {
        'crop_type': batch['crop_type'],
        'quantity': batch['quantity'],
        'shelf_life_percent': int(shelf_life_percent),
        'status': status,
        'current_price': 2450,
        'inventory_value': batch['quantity'] * 2450,
        'recommendations': f'Days stored: {days_stored}/{max_days}. ' + 
                          ('Monitor and continue storing.' if status == 'GREEN' else 'Consider selling soon.'),
        'temperature': batch['temperature'],
        'humidity': batch['humidity'],
        'storage_type': batch['storage_type']
    }
    
    return render_template('storage_dashboard.html', data=dashboard_data, batch_id=batch_id)


@app.route('/knowledge-hub', methods=['GET'])
def knowledge_hub():
    filter_state = request.args.get('state', '')
    filter_district = request.args.get('district', '')
    filter_commodity = request.args.get('commodity', '')
    
    latest_prices = []
    top_commodities = []
    all_commodities_data = []
    states_list = []
    districts_list = []
    commodities_list = []
    
    if DATASETS_LOADED and not market_prices_df.empty:
        try:
            states_list = sorted(market_prices_df['State'].unique().tolist())
            commodities_list = sorted(market_prices_df['Commodity'].unique().tolist())
            
            filtered_df = market_prices_df.copy()
            
            if filter_state:
                filtered_df = filtered_df[filtered_df['State'] == filter_state]
                districts_list = sorted(filtered_df['District'].unique().tolist())
            
            if filter_district:
                filtered_df = filtered_df[filtered_df['District'] == filter_district]
            
            if filter_commodity:
                filtered_df = filtered_df[filtered_df['Commodity'].str.contains(filter_commodity, case=False, na=False)]
            
            if not filtered_df.empty:
                latest_prices = filtered_df.head(100).to_dict('records')
            
            if not market_prices_df.empty:
                top_priced = market_prices_df.nlargest(10, 'Modal_Price')
                top_commodities = top_priced[['Commodity', 'Modal_Price', 'State', 'Market']].to_dict('records')
            
            commodity_stats = market_prices_df.groupby('Commodity').agg({
                'Modal_Price': ['mean', 'min', 'max', 'count']
            }).reset_index()
            commodity_stats.columns = ['Commodity', 'AvgPrice', 'MinPrice', 'MaxPrice', 'RecordCount']
            commodity_stats = commodity_stats.sort_values('AvgPrice', ascending=False)
            all_commodities_data = commodity_stats.to_dict('records')
            
        except Exception as e:
            print(f"Error in knowledge hub: {e}")
    
    return render_template('knowledge_hub.html',
                          latest_prices=latest_prices,
                          top_commodities=top_commodities,
                          all_commodities_data=all_commodities_data,
                          states_list=states_list,
                          districts_list=districts_list,
                          commodities_list=commodities_list,
                          filter_state=filter_state,
                          filter_district=filter_district,
                          filter_commodity=filter_commodity,
                          data_loaded=DATASETS_LOADED,
                          total_records=len(latest_prices))


@app.route('/api/commodity-trend/<commodity>')
def commodity_trend(commodity):
    import json
    
    if not DATASETS_LOADED or market_prices_df.empty:
        return json.dumps({'error': 'Data not loaded'}), 404
    
    try:
        commodity_data = market_prices_df[market_prices_df['Commodity'].str.lower() == commodity.lower()]
        
        if commodity_data.empty:
            return json.dumps({'error': 'Commodity not found'}), 404
        
        state_prices = commodity_data.groupby('State').agg({
            'Modal_Price': 'mean',
            'Min_Price': 'mean',
            'Max_Price': 'mean'
        }).reset_index()
        state_prices = state_prices.sort_values('Modal_Price', ascending=False).head(15)
        
        result = {
            'commodity': commodity,
            'states': state_prices['State'].tolist(),
            'modal_prices': state_prices['Modal_Price'].round(2).tolist(),
            'min_prices': state_prices['Min_Price'].round(2).tolist(),
            'max_prices': state_prices['Max_Price'].round(2).tolist(),
            'overall_avg': float(commodity_data['Modal_Price'].mean().round(2)),
            'overall_min': float(commodity_data['Modal_Price'].min().round(2)),
            'overall_max': float(commodity_data['Modal_Price'].max().round(2)),
            'total_records': len(commodity_data)
        }
        
        return json.dumps(result), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return json.dumps({'error': str(e)}), 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
