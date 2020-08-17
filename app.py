import streamlit as st

#EDA pkg
import pandas as pd
import numpy as np

# Model Load/Save
from joblib import load
import joblib
import os

#Data Viz pkg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Database
import sqlite3

class Monitor(object):
	"""docstring for Monitor"""

	conn = sqlite3.connect('data.db')
	c = conn.cursor()

	def __init__(self,age=None ,workclass=None ,fnlwgt=None ,education=None ,education_num=None ,marital_status=None ,occupation=None ,relationship=None ,race=None ,sex=None ,capital_gain=None ,capital_loss=None ,hours_per_week=None ,native_country=None,predicted_class=None,model_class=None):
		super(Monitor, self).__init__()
		self.age = age
		self.workclass = workclass
		self.fnlwgt = fnlwgt
		self.education = education
		self.education_num = education_num
		self.marital_status = marital_status
		self.occupation = occupation
		self.relationship = relationship
		self.race = race
		self.sex = sex
		self.capital_gain = capital_gain
		self.capital_loss = capital_loss
		self.hours_per_week = hours_per_week
		self.native_country = native_country
		self.predicted_class = predicted_class
		self.model_class = model_class

	def __repr__(self):
		# return "Monitor(age ={self.age},workclass ={self.workclass},fnlwgt ={self.fnlwgt},education ={self.education},education_num ={self.education_num},marital_status ={self.marital_status},occupation ={self.occupation},relationship ={self.relationship},race ={self.race},sex ={self.sex},capital_gain ={self.capital_gain},capital_loss ={self.capital_loss},hours_per_week ={self.hours_per_week},native_country ={self.native_country},predicted_class ={self.predicted_class},model_class ={self.model_class})".format(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class)
		"Monitor(age = {self.age},workclass = {self.workclass},fnlwgt = {self.fnlwgt},education = {self.education},education_num = {self.education_num},marital_status = {self.marital_status},occupation = {self.occupation},relationship = {self.relationship},race = {self.race},sex = {self.sex},capital_gain = {self.capital_gain},capital_loss = {self.capital_loss},hours_per_week = {self.hours_per_week},native_country = {self.native_country},predicted_class = {self.predicted_class},model_class = {self.model_class})".format(self=self)

	def create_table(self):
		self.c.execute('CREATE TABLE IF NOT EXISTS salaryprediction(age NUMERIC,workclass NUMERIC,fnlwgt NUMERIC,education NUMERIC,education_num NUMERIC,marital_status NUMERIC,occupation NUMERIC,relationship NUMERIC,race NUMERIC,sex NUMERIC,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,native_country NUMERIC,predicted_class NUMERIC,model_class TEXT)')

	def add_data(self):
		self.c.execute('INSERT INTO salaryprediction(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,predicted_class,model_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class))
		self.conn.commit()

	def view_all_data(self):
		self.c.execute('SELECT * FROM salaryprediction')
		data = self.c.fetchall()
		# for row in data:
		# 	print(row)
		return data



		


# Functions

## Load css
def load_css(css_name):
	with open(css_name) as f:
		st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

## Load icon
def load_icon(name):
	st.markdown('<i class ="material-icons">{}</i>'.format(name), unsafe_allow_html=True)

## remote_css
def remote_css(url):
    st.markdown('<style src="{}"></style>'.format(url), unsafe_allow_html=True)

## icon-css
def icon_css(icone_name):
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

## Getting value from dictionary
def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value

## Getting Keys from dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key

## Load Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# Main
def main():
	""" Salary Prediction ML App with Streamlit """
	st.title("Salary Prediction")
	st.text("ML Prediction App with Streamlit")

	# Loading Dataset
	df = pd.read_csv('data/adult_salary.csv')
	df2 = pd.read_csv('data/adult_salary_data.csv')

	# Sidebar (TABS/ Menus)
	bars = ['EDA','Countries','Prediction','Monitor','About']
	choice = st.sidebar.selectbox("Choose Activity", bars)


	# Choice EDA
	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")
		load_css('icon.css') #function defines at the top
		load_icon('dashboard') #function defines at the top

		if st.checkbox("Show Dataset Preview"):
			num = st.number_input("Enter Number of Rows to Preview: ", value=5)
			st.dataframe(df.head(num))

		if st.checkbox("Shape of Dataset"):
			st.write(df.shape)
			dim = st.radio("Show Dimensions by :",("Rows","Columns"))

			if dim == "Rows":
				st.text("Number of Rows :")
				st.write(df.shape[0])
			elif dim == "Columns":
				st.text("Number of Columns :")
				st.write(df.shape[1])

		if st.checkbox("Column Names"):
			all_columns = df.columns.tolist()
			st.write(all_columns)

		if st.checkbox("Select Columns to Show"):
			all_columns = df.columns.tolist()
			selected_col = st.multiselect("Select Columns", all_columns)
			new_col = df[selected_col]
			st.dataframe(new_col)
		
		if st.checkbox("Select Rows to Show"):
			selected_index = st.multiselect("Select Rows: ", df.head(15).index)
			selected_row = df.loc[selected_index]
			st.dataframe(selected_row)

		if st.checkbox("Show Info"):
			st.write(df.dtypes)

		if st.checkbox("Show Description"):
			st.write(df.describe())

		if st.checkbox("Value Counts"):
			st.text("Value count of Target/ Class")
			st.write(df.iloc[:,-1].value_counts())

		st.subheader("Data Visualization")
		load_css('icon.css')
		load_icon('show_charts')

		# Correlation plot with Matplotlib
		if st.checkbox("Correlation Plot [using Matplotlib]"):
			plt.matshow(df.corr())
			st.pyplot()

		# Correlation plot with Seaborn
		if st.checkbox("Correlation Plot with Annotation [using Seaborn]"):
			st.write(sns.heatmap(df.corr(), annot=True))
			st.pyplot()

	
	# CHOICE FOR COUNTRIES
	if choice == 'Countries':
		st.markdown("#### Demographics")
		d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
		selected_countries = st.selectbox("Select Country",tuple(d_native_country.keys()))
		st.text(selected_countries)

		result_df = df2[df2['native-country'].str.contains(selected_countries)]
		st.dataframe(result_df)
	

		if st.checkbox("Select Columns To Show"):
				result_df_columns = result_df.columns.tolist()
				selected_columns = st.multiselect('Select',result_df_columns)
				new_df = df2[selected_columns]
				st.dataframe(new_df)

				if st.checkbox("Plot"):
					st.area_chart(df[selected_columns])
					st.pyplot()

	# Choice Prediction
	if choice == 'Prediction':
		st.subheader("Prediction of Salary")
		st.markdown('<style>' + open('icon.css').read() + '</style>', unsafe_allow_html=True)
		st.markdown('<i class="material-icons">mood</i>', unsafe_allow_html=True)
		
		load_css('icon.css') # function defines at the top
		# load_icon('timeline') #function defines at the top	

		# Dictionary of Mapped Values (mapped_dict.txt)
		d_workclass = {"Never-worked": 0, "Private": 1, "Federal-gov": 2, "?": 3, "Self-emp-inc": 4, "State-gov": 5, "Local-gov": 6, "Without-pay": 7, "Self-emp-not-inc": 8}
		#d_education = {"Some-college": 0, "10th": 1, "Doctorate": 2, "1st-4th": 3, "12th": 4, "Masters": 5, "5th-6th": 6, "9th": 7, "Preschool": 8, "HS-grad": 9, "Assoc-acdm": 10, "Bachelors": 11, "Prof-school": 12, "Assoc-voc": 13, "11th": 14, "7th-8th": 15}
		d_education = {"1st-4th": 3, "5th-6th": 6, "7th-8th": 15,"9th": 7, "10th": 1, "11th": 14, "12th": 4,"Preschool": 8,"Prof-school": 12, "Some-college": 0,"HS-grad": 9,"Bachelors": 11,"Masters": 5,"Assoc-acdm": 10,  "Assoc-voc": 13, "Doctorate": 2}
		d_marital_status = {"Separated": 0, "Married-spouse-absent": 1, "Married-AF-spouse": 2, "Married-civ-spouse": 3, "Never-married": 4, "Widowed": 5, "Divorced": 6}
		d_occupation = {"Tech-support": 0, "Farming-fishing": 1, "Prof-specialty": 2, "Sales": 3, "?": 4, "Transport-moving": 5, "Armed-Forces": 6, "Other-service": 7, "Handlers-cleaners": 8, "Exec-managerial": 9, "Adm-clerical": 10, "Craft-repair": 11, "Machine-op-inspct": 12, "Protective-serv": 13, "Priv-house-serv": 14}
		d_relationship = {"Other-relative": 0, "Not-in-family": 1, "Own-child": 2, "Wife": 3, "Husband": 4, "Unmarried": 5}
		d_race = {"Amer-Indian-Eskimo": 0, "Black": 1, "White": 2, "Asian-Pac-Islander": 3, "Other": 4}
		d_sex = {"Female": 0, "Male": 1}
		d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
		d_class = {">50K": 0, "<=50K": 1}

		# Countries Abbrevation (country_abbrev.txt)

		countries_flag = {'af': 'Afghanistan','al': 'Albania','dz': 'Algeria','as': 'American Samoa','ad': 'Andorra','ao': 'Angola','ai': 'Anguilla','aq': 'Antarctica','ag': 'Antigua And Barbuda','ar': 'Argentina','am': 'Armenia','aw': 'Aruba','au': 'Australia','at': 'Austria','az': 'Azerbaijan','bs': 'Bahamas','bh': 'Bahrain','bd': 'Bangladesh','bb': 'Barbados','by': 'Belarus','be': 'Belgium','bz': 'Belize','bj': 'Benin','bm': 'Bermuda','bt': 'Bhutan','bo': 'Olivia','ba': 'Bosnia And Herzegovina','bw': 'Botswana','bv': 'Bouvet Island','br': 'Brazil','io': 'British Indian Ocean Territory','bn': 'Brunei Darussalam','bg': 'Bulgaria','bf': 'Burkina Faso','bi': 'Burundi','kh': 'Cambodia','cm': 'Cameroon','ca': 'Canada','cv': 'Cape Verde','ky': 'Cayman Islands','cf': 'Central African Republic','td': 'Chad','cl': 'Chile','cn': "People'S Republic Of China",'cx': 'Hristmas Island','cc': 'Cocos (Keeling) Islands','co': 'Colombia','km': 'Comoros','cg': 'Congo','cd': 'Congo, The Democratic Republic Of','ck': 'Cook Islands','cr': 'Costa Rica','ci': "Côte D'Ivoire",'hr': 'Croatia','cu': 'Cuba','cy': 'Cyprus','cz': 'Czech Republic','dk': 'Denmark','dj': 'Djibouti','dm': 'Dominica','do': 'Dominican Republic','ec': 'Ecuador','eg': 'Egypt','eh': 'Western Sahara','sv': 'El Salvador','gq': 'Equatorial Guinea','er': 'Eritrea','ee': 'Estonia','et': 'Ethiopia','fk': 'Falkland Islands (Malvinas)','fo': 'Aroe Islands','fj': 'Fiji','fi': 'Finland','fr': 'France','gf': 'French Guiana','pf': 'French Polynesia','tf': 'French Southern Territories','ga': 'Gabon','gm': 'Gambia','ge': 'Georgia','de': 'Germany','gh': 'Ghana','gi': 'Gibraltar','gr': 'Greece','gl': 'Greenland','gd': 'Grenada','gp': 'Guadeloupe','gu': 'Guam','gt': 'Guatemala','gn': 'Guinea','gw': 'Guinea-Bissau','gy': 'Guyana','ht': 'Haiti','hm': 'Heard Island And Mcdonald Islands','hn': 'Honduras','hk': 'Hong Kong','hu': 'Hungary','is': 'Iceland','in': 'India','id': 'Indonesia','ir': 'Iran, Islamic Republic Of','iq': 'Iraq','ie': 'Ireland','il': 'Israel','it': 'Italy','jm': 'Jamaica','jp': 'Japan','jo': 'Jordan','kz': 'Kazakhstan','ke': 'Kenya','ki': 'Kiribati','kp': "Korea, Democratic People'S Republic Of",'kr': 'Korea, Republic Of','kw': 'Kuwait','kg': 'Kyrgyzstan','la': "Lao People'S Democratic Republic",'lv': 'Latvia','lb': 'Lebanon','ls': 'Lesotho','lr': 'Liberia','ly': 'Libyan Arab Jamahiriya','li': 'Liechtenstein','lt': 'Lithuania','lu': 'Luxembourg','mo': 'Macao','mk': 'Macedonia, The Former Yugoslav Republic Of','mg': 'Madagascar','mw': 'Malawi','my': 'Malaysia','mv': 'Maldives','ml': 'Mali','mt': 'Malta','mh': 'Marshall Islands','mq': 'Martinique','mr': 'Mauritania','mu': 'Mauritius','yt': 'Mayotte','mx': 'Mexico','fm': 'Micronesia, Federated States Of','md': 'Moldova, Republic Of','mc': 'Monaco','mn': 'Mongolia','ms': 'Montserrat','ma': 'Morocco','mz': 'Mozambique','mm': 'Myanmar','na': 'Namibia','nr': 'Nauru','np': 'Nepal','nl': 'Netherlands','an': 'Netherlands Antilles','nc': 'New Caledonia','nz': 'New Zealand','ni': 'Nicaragua','ne': 'Niger','ng': 'Nigeria','nu': 'Niue','nf': 'Norfolk Island','mp': 'Northern Mariana Islands','no': 'Norway','om': 'Oman','pk': 'Pakistan','pw': 'Palau','ps': 'Palestinian Territory, Occupied','pa': 'Panama','pg': 'Papua New Guinea','py': 'Paraguay','pe': 'Peru','ph': 'Philippines','pn': 'Pitcairn','pl': 'Poland','pt': 'Portugal','pr': 'Puerto Rico','qa': 'Qatar','re': 'Réunion','ro': 'Romania','ru': 'Russian Federation','rw': 'Rwanda','sh': 'Saint Helena','kn': 'Saint Kitts And Nevis','lc': 'Saint Lucia','pm': 'Saint Pierre And Miquelon','vc': 'Saint Vincent And The Grenadines','ws': 'Samoa','sm': 'San Marino','st': 'Sao Tome And Principe','sa': 'Saudi Arabia','sn': 'Senegal','cs': 'Serbia And Montenegro','sc': 'Seychelles','sl': 'Sierra Leone','sg': 'Singapore','sk': 'Slovakia','si': 'Slovenia','sb': 'Solomon Islands','so': 'Somalia','za': 'South Africa','gs': 'South Georgia And South Sandwich Islands','es': 'Spain','lk': 'Sri Lanka','sd': 'Sudan','sr': 'Suriname','sj': 'Svalbard And Jan Mayen','sz': 'Swaziland','se': 'Sweden','ch': 'Switzerland','sy': 'Syrian Arab Republic','tw': 'Taiwan, Republic Of China','tj': 'Tajikistan','tz': 'Tanzania, United Republic Of','th': 'Thailand','tl': 'Timor-Leste','tg': 'Togo','tk': 'Tokelau','to': 'Tonga','tt': 'Trinidad And Tobago','tn': 'Tunisia','tr': 'Turkey','tm': 'Turkmenistan','tc': 'Turks And Caicos Islands','tv': 'Tuvalu','ug': 'Uganda','ua': 'Ukraine','ae': 'United Arab Emirates','gb': 'United Kingdom','us': 'United States','um': 'United States Minor Outlying Islands','uy': 'Uruguay','uz': 'Uzbekistan','ve': 'Venezuela','vu': 'Vanuatu','vn': 'Viet Nam','vg': 'British Virgin Islands','vi': 'U.S. Virgin Islands','wf': 'Wallis And Futuna','ye': 'Yemen','zw': 'Zimbabwe'}


		# Collection of User Input
		load_icon('date_range')
		age = st.slider("Select Age",16,90)
		load_icon('work')
		workclass = st.selectbox("Select Work Class", tuple(d_workclass.keys()))
		fnlwgt = int(st.number_input("Enter FNLWGT",12285,1484705))
		load_icon('school')
		education = st.selectbox("Select Education",tuple(d_education.keys()))
		education_num = st.slider("Select Education Level",1,16)
		marital_status = st.selectbox("Select Marital-status",tuple(d_marital_status.keys()))
		load_icon('work')
		occupation = st.selectbox("Select Occupation",tuple(d_occupation.keys()))
		relationship = st.selectbox("Select Relationship",tuple(d_relationship.keys()))
		race = st.selectbox("Select Race",tuple(d_race.keys()))
		load_icon('people')
		sex = st.radio("Select Sex",tuple(d_sex.keys()))
		st.text('Money')
		load_icon('attach_money')
		load_icon('insert_emoticon')
		capital_gain = st.number_input("Capital Gain",0,99999,value=1)
		load_icon('mood_bad')
		capital_loss = st.number_input("Capital Loss",0,4356,value=1)
		load_icon('access_time')
		hours_per_week = st.number_input("Hours Per Week ",0,99,value=1)
		load_icon('place')
		native_country = st.selectbox("Select Native Country",tuple(d_native_country.keys()))


		# Getting values for each input Key
		k_workclass = get_value(workclass,d_workclass)
		k_education = get_value(education,d_education)
		k_marital_status = get_value(marital_status,d_marital_status)
		k_occupation = get_value(occupation,d_occupation)
		k_relationship = get_value(relationship,d_relationship)
		k_race = get_value(race,d_race)
		k_sex = get_value(sex,d_sex)
		k_native_country = get_value(native_country,d_native_country)

		# Showing Selected Entered information to User
		## Show Summary
		selected_columns = [age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country]
		vectorized_result = [age ,k_workclass ,fnlwgt ,k_education ,education_num ,k_marital_status ,k_occupation ,k_relationship ,k_race ,k_sex ,capital_gain ,capital_loss ,hours_per_week ,k_native_country]
		vectorization = np.array(vectorized_result).reshape(1,-1)
			
		if st.checkbox("Show Summary"):
			st.info (selected_columns)

			if st.checkbox("Summary in JSON Format"):
				simple_result = {"age":age,
				"workclass":workclass,
				"fnlwgt":fnlwgt,
				"education":education,
				"education_num":education_num,
				"marital_status":marital_status,
				"occupation":occupation,
				"relationship":relationship,
				"race":race,
				"sex":sex,
				"capital_gain":capital_gain,
				"capital_loss":capital_loss,
				"hours_per_week":hours_per_week,
				"native_country":native_country}

				st.subheader("Pretified Result in JSON")
				st.json(simple_result)

			if st.checkbox("Summary in Encoded Format"):
				st.text("Using Encoding for Prediction")
				st.success(vectorized_result)


		# Making Predictions
		st.subheader("Prediction")
		if st.checkbox("Make Prediction"):
			
			all_ml_dict = {'LR':"LogisticRegression",
				'RForest':"RandomForestClassifier",
				'NB':"MultinomialNB"}

			# Model Selection
			model_choice = st.selectbox('Model Choice',list(all_ml_dict.keys()))
			prediction_label = {">50K": 0, "<=50K": 1}

			if st.button("Predict"):
				if model_choice =='LR':
					model_predictor = load_model("models/salary_logit_model.pkl")
					prediction = model_predictor.predict(vectorization)
				elif model_choice =='RForest':
					model_predictor = load_model("models/salary_rf_model.pkl")
					prediction = model_predictor.predict(vectorization)
				elif model_choice =='NB':
					model_predictor = load_model("models/salary_nv_model.pkl")
					prediction = model_predictor.predict(vectorization)
				
				final_outcome = get_key(prediction,prediction_label)
				
				# Adding Data into the database (Monitor class defined at the top)
				monitor = Monitor(age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country,final_outcome,model_choice)
				monitor.create_table()
				monitor.add_data()

				st.success("Predicted Salary as :: {}".format(final_outcome))
				
	# CHOICE is Monitor
	if choice == 'Monitor':
		st.subheader("Metrics of Predictions")
		# Create your connection.
		conn = sqlite3.connect('data.db')

		fetched_data = pd.read_sql_query("SELECT * FROM salaryprediction", conn)
		st.dataframe(fetched_data)

	# ABOUT CHOICE
	if choice == 'About':
		st.subheader("About")
		st.markdown("""
			#### Salary Predictor ML App
			##### Built with Streamlit

			#### By
			+ Hrishikesh Sharad Malkar
			+ References: Jesus Saves@[JCharisTech](https://jcharistech.com)

			""")


if __name__ == '__main__':
	main()
