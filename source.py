#IMPORTING NECESSARY PACKAGES
import base64

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import base64
from sklearn.tree import DecisionTreeRegressor
#CREATING A STR
# EAMLIT PAGE
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp{
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg('bg.jpg')
with open("scratch.css") as sd:
    st.markdown(f"<style>{sd.read()}</style>",unsafe_allow_html=True)
st.title('BHARATH KISAN HELPLINE')
st.sidebar.title('CHOOSE A FACILITY')

with st.sidebar:
    option = option_menu('',options=('CROP RECOMMENDER','YIELD PREDICTOR','PROFIT PREDICTOR','DISEASE ENCYCLOPEDIA'),
                         icons=['tree-fill','flower1','currency-exchange','book'])

if option == 'CROP RECOMMENDER':
    # DATA PREPROCESSING
    dataset1 = pd.read_csv('crop.csv')
    dataset1['label'] = dataset1['label'].map({'rice': '1',
                                               'maize': '2',
                                               'chickpea': '3',
                                               'kidneybeans': '4',
                                               'pigeonpeas': '5',
                                               'mothbeans': '6'})
    # TRAINING THE MODEL
    x1 = dataset1.iloc[:, :-1].values
    y1 = dataset1.iloc[:, -1].values
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.25, random_state=0)
    lr1 = SVC()
    lr2 = KNeighborsClassifier()
    lr3= GaussianNB()
    lr1.fit(x1_train, y1_train)
    lr2.fit(x1_train, y1_train)
    lr3.fit(x1_train, y1_train)
    lr1_pred = lr1.predict(x1_test)
    lr2_pred = lr2.predict(x1_test)
    lr3_pred = lr3.predict(x1_test)

    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input('Nitrogen in mg/Kg', step=1,max_value=100,min_value=10)
    with col2:
        p = st.number_input('Phosphurus  in mg/Kg', step=1,max_value=100,min_value=10)
    with col3:
        k = st.number_input('Potassium in mg/Kg', step=1,max_value=50,min_value=0)
    with col1:
        temp = st.number_input('Temperature IN Deg(C)', min_value=5,max_value=45)
    with col2:
        ph = st.number_input('ENTER PH VALUE')
        humidity = st.number_input('ENTER HUMIDITY IN g/m^3 ')
    with col3:
        rainfall = st.number_input('ENTER RAINFALL IN mm')
    with col2:
        sub1 = st.button('RECOMMEND THE BEST CROP FOR ME')

    new1 = [[n, p, k, temp, ph, humidity, rainfall]]

    if sub1:
        dict = {'1': 'White Rice : Oryza sativa', '2': 'Maize : Zea mays', '3': 'Chick peas : Cicer arietinum',
                '4': 'Kidney beans : Phaseolus vulgaris', '5': 'Pigeon peas : Cajanus cajan',
                '6': 'Moth beans: Vigna aconitifolia'}

        lr1_res = lr1.predict(new1)
        lr1_res1 = lr2.predict(new1)
        lr1_res2 = lr3.predict(new1)
        # info = dict[int(lr1_res)]
        st.header('THE BEST CROP RECOMMENDED FOR YOU BY THE MODEL IS(svm) : ')
        st.success(dict[lr1_res[0]])
        st.header('THE BEST CROP RECOMMENDED FOR YOU BY THE MODEL IS(knn) : ')
        st.success(dict[lr1_res1[0]])
        st.header('THE BEST CROP RECOMMENDED FOR YOU BY THE MODEL IS(nvb) : ')
        st.success(dict[lr1_res2[0]])
        print('MODEL PARAMETERS')
        print("ACCURACY (svm) : {0}%".format(round(accuracy_score(y1_test, lr1_pred), 2) * 100))
        print("ACCURACY (knn) : {0}%".format(round(accuracy_score(y1_test, lr2_pred),2)* 100))
        print("ACCURACY (nvb) : {0}%".format(round(accuracy_score(y1_test, lr3_pred),2)*100))
        print("MICRO PRECISION : {0}%".format(round(precision_score(y1_test, lr1_pred, average='micro'), 2) * 100))
        print("MACRO PRECISION  : {0}%".format(round(f1_score(y1_test, lr1_pred, average='macro'), 2) * 100))
        print("WEIGHTED PRECISION  : {0}%".format(round(recall_score(y1_test, lr1_pred, average='weighted'), 2) * 100))

elif option == 'YIELD PREDICTOR':
    dataset2 = pd.read_csv("YIELD & PROFIT.csv")
    print(dataset2['Crop'].unique())
    # mapping state name
    a1 = dataset2['State_Name'].unique().tolist()
    b1 = np.arange(1, len(a1) + 1, 1)
    c1 = {i: j for i, j in zip(a1, b1)}
    dataset2['State_Name'] = dataset2['State_Name'].map(c1)
    print(c1)
    # mapping district name
    a2 = dataset2['District_Name'].unique().tolist()
    b2 = np.arange(1, len(a2) + 1, 1)
    c2 = {i: j for i, j in zip(a2, b2)}
    dataset2['District_Name'] = dataset2['District_Name'].map(c2)
    print(c2)
    # mapping season
    a3 = dataset2['Season'].unique().tolist()
    b3 = np.arange(1, len(a3) + 1, 1)
    c3 = {i: j for i, j in zip(a3, b3)}
    dataset2['Season'] = dataset2['Season'].map(c3)
    print(c3)
    # mapping crop
    a4 = dataset2['Crop'].unique().tolist()
    b4 = np.arange(1, len(a4) + 1, 1)
    c4 = {i: j for i, j in zip(a4, b4)}
    dataset2['Crop'] = dataset2['Crop'].map(c4)

    dataset2 = dataset2.dropna()

    col1, col2, col3 = st.columns(3)
    with col1:
        # CREATING INPUT VARIABLES IN STREAMLIT PAGE
        state1 = st.selectbox('ENTER STATE NAME : ', options=c1)
        state1 = c1[state1]
    with col2:
        district1 = st.selectbox('ENTER DISTRICT NAME : ', options=c2)
        district1 = c2[district1]
    with col3:
        crop_year1 = st.number_input('ENTER CROP YEAR : ', min_value=2015, step=1)
        season1 = st.selectbox('ENTER SEASON : ', options=c3)
    with col1:
        season1 = c3[season1]
        area1 = st.number_input('ENTER CULTIVATING AREA in acres : ', min_value=1, step=1)
    with col2:
        crop1 = st.selectbox('ENTER CROP : ', options=c4)
        crop1 = c4[crop1]
        sub2 = st.button('PREDICT PRODUCTION OF CROP')

    new2 = [[state1, district1, crop_year1, season1, crop1, area1]]


    # TRAINING THE MODEL & INTEGRATING WITH STREAMLIT PAGE
    if sub2:
        x2 = dataset2.iloc[:, :6].values
        y2 = dataset2.iloc[:, 6].values
        x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.1, random_state=11000)
        lr = DecisionTreeRegressor()
        lr.fit(x2_train, y2_train)
        lr_pred = lr.predict(x2_test)
        lr_res = lr.predict(new2)

        st.header('PREDICTED GROSS yeild in tons: ')
        st.success((int(lr_res[0])))

        # st.header('MODEL PARAMETERS')
        print("MAE : {0}".format(round(mean_absolute_error(y2_test, lr_pred), 2)))
        print("MSE : {0}".format(round(mean_squared_error(y2_test, lr_pred), 4)))
        print("RMSE  : {0}".format(round(np.sqrt(mean_squared_error(y2_test, lr_pred)), 2)))
        print("R2 score  : {0}".format(r2_score(y2_test, lr_pred) * 100))
        st.header('REPORT')
        report1 = dataset2.describe()
        report1_df = pd.DataFrame(report1).transpose()
        st.dataframe(report1_df, height=212, width=750)
    # else:
    #     if
    #     st.error('CHECK ALL INPUTS & CLICK SUBMIT')
elif option == 'PROFIT PREDICTOR':
    dataset2 = pd.read_csv("YIELD & PROFIT.csv")
    dataset4 = pd.read_csv("YIELD & PROFIT.csv")
    # mapping state name
    a1 = dataset2['State_Name'].unique().tolist()
    b1 = np.arange(1, len(a1) + 1, 1)
    c1 = {i: j for i, j in zip(a1, b1)}
    dataset2['State_Name'] = dataset2['State_Name'].map(c1)
    dataset4['State_Name'] = dataset4['State_Name'].map(c1)
    print(c1)
    # mapping district name
    a2 = dataset2['District_Name'].unique().tolist()
    b2 = np.arange(1, len(a2) + 1, 1)
    c2 = {i: j for i, j in zip(a2, b2)}
    dataset2['District_Name'] = dataset2['District_Name'].map(c2)
    dataset4['District_Name'] = dataset4['District_Name'].map(c2)
    print(c2)
    # mapping season
    a3 = dataset2['Season'].unique().tolist()
    b3 = np.arange(1, len(a3) + 1, 1)
    c3 = {i: j for i, j in zip(a3, b3)}
    dataset2['Season'] = dataset2['Season'].map(c3)
    dataset4['Season'] = dataset4['Season'].map(c3)
    print(c3)
    # mapping crop
    a4 = dataset2['Crop'].unique().tolist()
    b4 = np.arange(1, len(a4) + 1, 1)
    c4 = {i: j for i, j in zip(a4, b4)}
    dataset2['Crop'] = dataset2['Crop'].map(c4)
    dataset4['Crop'] = dataset4['Crop'].map(c4)

    dataset2 = dataset2.dropna()
    dataset4 = dataset4.dropna()

    dataset2.loc[dataset2['Profit'] <= 0, 'result'] = '0'
    dataset2.loc[dataset2['Profit'] > 0, 'result'] = '1'

    x3 = dataset2.iloc[:, :6].values
    y3 = dataset2.iloc[:, -1].values

    x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=50000)
    sc = StandardScaler()
    x3_train = sc.fit_transform(x3_train)
    x3_test = sc.transform(x3_test)
    knn = KNeighborsClassifier()
    knn.fit(x3_train, y3_train)
    knn_pred = knn.predict(x3_test)

    dict2 = {'1': 'YOU HAVE CHOSEN A PROFITABLE CROP', '0': 'YOU HAVE CHOSEN A LOSSY CROP'}

    # CREATING INPUT VARIABLES IN STREAMLIT PAGE
    col1, col2, col3 = st.columns(3)
    with col1:
        state2 = st.selectbox('ENTER STATE NAME ', options=c1)
        state2 = c1[state2]
    with col2:
        district2 = st.selectbox('ENTER DISTRICT NAME', options=c2)
        district2 = c2[district2]
    with col3:
        crop_year2 = st.number_input('ENTER CROP YEAR', min_value=1999, step=1)
        season2 = st.selectbox('ENTER SEASON', options=c3)
    with col1:
        season2 = c3[season2]
        area2 = st.number_input('ENTER CULTIVATING AREA in acres', min_value=1, step=1)
    with col2:
        crop2 = st.selectbox('ENTER CROP', options=c4)
        crop2 = c4[crop2]
        sub3 = st.button('PREDICT PROFITABILITY OF MY CROP')

    new3 = [[state2, district2, crop_year2, season2, crop2, area2]]

    x4 = dataset4.iloc[:, :6].values
    y4 = dataset4.iloc[:, -1].values
    x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.1, random_state=11000)
    lr = DecisionTreeRegressor()
    lr.fit(x4_train, y4_train)
    lr_pred1 = lr.predict(x4_test)
    lr_res2 = lr.predict(new3)



    # TRAINING THE MODEL & INTEGRATING WITH STREAMLIT PAGE
    if sub3:
        knn_res = knn.predict(sc.transform(new3))
        st.header('PREDICTED GROSS YIELD : ')
        if(dict2[knn_res[0]]=='YOU HAVE CHOSEN A LOSSY CROP'):
            st.error(dict2[knn_res[0]])
            st.error('Rs ' + str(int(lr_res2[0])))
        else:
            st.success(dict2[knn_res[0]])
            st.success('Rs ' + str(int(lr_res2[0])))
        st.header('MODEL PARAMETERS')
        st.success("ACCURACY : {0}%".format(round(accuracy_score(y3_test, knn_pred), 2) * 100))
        print("PRECISION : {0}%".format(round(precision_score(y3_test, knn_pred, pos_label='1'), 2) * 100))
        print("F1 SCORE  : {0}%".format(round(f1_score(y3_test, knn_pred, pos_label='1'), 2) * 100))
        print("R2 SCORE  : {0}%".format(round(recall_score(y3_test, knn_pred, pos_label='1'), 2) * 100))

    else:
        if area2 == 0:
            st.error('CHECK ALL INPUTS & CLICK SUBMIT')
else:
    disease = st.selectbox('SELECT A CROP',options={'TOMATO','POTATO','COTTON','PUMPKIN','CABBAGE'})
#COMPLETE TOMATO DATABASE
    if disease == 'TOMATO':
        tomato_disease = st.selectbox('SELECT A TOMATO DISEASE', options={'Alternaria Canker',
                                                                   'Bacterial Canker',
                                                                   'Bacterial Speck',
                                                                   'Bacterial Spot',
                                                                   'Early Blight',
                                                                   'Gray Leaf Spot',
                                                                   'Late Blight'})
        sub2 = st.button('SUBMIT')

        if sub2:
            if tomato_disease == 'Alternaria Canker':
                with st.container():
                    p11 = Image.open('pics/1 tomato/1 Alternaria Canke.jpg')
                    st.image(p11, caption='fig. Alternaria Canker')

                    st.write(
                        'Symptoms : Symptoms of Alternaria canker appear on stems, leaves and fruit. Large areas of the leaf lamina between veins is killed, leading to leaf curling and eventual death of the entire leaf.')
                    st.write(
                        'Control : Fungus overwinters in crop residue and is easily spread by wind. Wounding of young plants (by mechanical damage or pruning) provides an entry site for infection. Furrow or drip irrigation is preferred over sprinkler irrigation. Preventative fungicide sprays may be required if a “zero tolerance” for defects production system is needed.')

            elif tomato_disease == 'Bacterial Canker':
                with st.container():
                    p12 = Image.open('pics/1 tomato/2 bac canke.jpg')
                    st.image(p12, caption='fig. Bacterial Canker')

                    st.write(
                        'Symptoms : Bacterial canker is characterized by wilting and eventual death of the lower leaves, with the leaves drying up while still attached to the stem. Vascular tissue is discolored, brown, or brownish-yellow, and a characteristic yellow slime can be squeezed from affected stems. The bacterium that causes this disorder may be seed or soil born.')
                    st.write(
                        'Control : Crop rotations and careful seed source selection are primary considerations. Seed beds in infected areas should be sterilized. Mechanical damage to the transplants (such as topping) spreads the disease.')

            elif tomato_disease == 'Bacterial Speck':
                with st.container():
                    p13 = Image.open('pics/1 tomato/3.jpg')
                    st.image(p13, caption='fig. Bacterial Speck')

                    st.write(
                        'Symptoms : Bacterial speck is widely distributed. Symptoms may appear on any plant part. Leaves of infected plants are covered by small, dark brown, irregular patches of necrotic tissue that are surrounded by yellow halos. Disease severity is increased by leaf wetness from sprinkler irrigation, rain, or heavy dews.')
                    st.write(
                        'Control : Minimize wetting of the leaves by using drip or furrow irrigation. Copper sprays provide effective control.')

            elif tomato_disease == 'Bacterial Spot':
                with st.container():
                    p14 = Image.open('pics/1 tomato/4.jpg')
                    st.image(p14, caption='fig. Bacterial Spot')

                    st.write(
                        'Symptoms : Dark brown water soaked spots appear on the leaves; later these spots become blackish, and eventually the affected tissue drops out leaving a hole in the leaf. Black, raised specks that later become scab-like spots appear at the same time on fruit.')
                    st.write(
                        'Control : Crop rotations and careful transplant selection are important. Copper sprays provide some control. Good sanitation practices including prompt plow-down of stubble and weed control help prevent the disease.')

            elif tomato_disease == 'Early Blight':
                with st.container():
                    p15 = Image.open('pics/1 tomato/4.jpg')
                    st.image(p15, caption='fig. Early Blight')

                    st.write(
                        'Symptoms : Leaf symptoms of early blight are large irregular patches of black, necrotic tissue surrounded by larger yellow areas. The leaf spots have a characteristic concentric banding appearance (oyster-shell or bull’s eye).')
                    st.write(
                        'Control : Minimize wetting of the leaves by using drip or furrow irrigation. Infection occurs rapidly during periods of warm, wet weather. Fungicide sprays control the disease effectively.')

            elif tomato_disease == 'Gray Leaf Spot':
                with st.container():
                    p16 = Image.open('pics/1 tomato/6.jpg')
                    st.image(p16, caption='fig. Gray Leaf Spot')

                    st.write(
                        'Symptoms : Small brownish-black specks first appear on undersides of leaves. These later develop into larger necrotic areas, and the tissue often falls out, leaving a shot hole type appearance. Spots may be surrounded by a yellow halo. Yellowing, leaf drop, and defoliation may occur in severe cases.')
                    st.write(
                        'Control : The fungus can survive from year to year on Solanaceous weeds, so weed control is important. Leaf moisture from rains or dew increases disease severity. Fungicides may be used as recommended. Many commercial varieties are resistant.')

            elif tomato_disease == 'Late Blight':
                with st.container():
                    p17 = Image.open('pics/1 tomato/7.jpg')
                    st.image(p17, caption='fig. Late Blight')

                    st.write(
                        'Symptoms : Lesions on leaves appear as large watersoaked areas, that eventually turn brown and papery. Fruit lesions are large irregular greenish-brown patches having a greasy rough appearance. Green to black irregular lesions are also present on the stems.')
                    st.write(
                        'Control : The fungus develops during periods of cool wet weather. Fungicide sprays as a preventative measure during these periods may be needed if the crop is being grown near large areas of tomato relatives (Solanaceous weeds, potatoes).')

        else:
            st.error('SELECT A CROP & DISEASE')

    # COMPLETE POTATO DATABASE
    elif disease == 'POTATO':
        potato_disease = st.selectbox('SELECT A POTATO DISEASE',options = {'Bacterial Wilt',
                                                                    'Septoria leaf spot',
                                                                    'Late blight',
                                                                    'Early blight',
                                                                    'Common scab',
                                                                    'Black canker'})
        sub3 = st.button('SUBMIT')

        if sub3:
            if potato_disease == 'Bacterial Wilt':
                with st.container():
                    p21 = Image.open('pics/2 potato/1.jpg')
                    st.image(p21, caption='fig. Bacterial wilt')

                    st.write(
                        'Symptoms : When a tuber is cut in half, black or brown rings will, however, be visible. If left for a while or squeezed, these rings will exude a thick white fluid.')
                    st.write(
                        'Control : Bacterial wilt can be controlled by exposing the seed tubers to hot air (112 ºF) with 75% relative humidity for 30 min')

            elif potato_disease == 'Septoria leaf spot':
                with st.container():
                    p22 = Image.open('pics/2 potato/2.jpg')
                    st.image(p22, caption='fig. Septoria leaf spot')

                    st.write(
                        'Symptoms : Small, round to irregular spots with a grey center and dark margin on leaves. Spots usually start on lower leaves and gradually advance upwards. Fruits are rarely attacked')
                    st.write(
                        'Control : Today, Septoria malagutii and other septoria disease are controlled with a number of different methods including the use of fungicides and cultural controls. Fungicides such as Fluazinam, used for controlling late blight of potato, Phytotophoria infestans, have proven to be effective.')

            elif potato_disease == 'Late blight':
                with st.container():
                    p23 = Image.open('pics/2 potato/3.jpg')
                    st.image(p23, caption='fig. Late blight')

                    st.write(
                        'Symptoms : Severe infections cause all foliage to rot, dry out and fall to the ground, stems to dry out and plants to die.Affected tubers display dry brown-colored spots on their skins and flesh. This disease acts very quickly. If it is not controlled, infected plants will die within two or three days.')
                    st.write(
                        'Control : Late blight is controlled by eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices, and applying fungicides when necessary. Air drainage to facilitate the drying of foliage each day is important.')

            elif potato_disease == 'Early blight':
                with st.container():
                    p24 = Image.open('pics/2 potato/4.jpeg')
                    st.image(p24, caption='fig. Early blight')

                    st.write(
                        'Symptoms : Tissue surrounding the spots may turn yellow. If high temperature and humidity occur at this time, much of the foliage is killed. Lesions on the stems are similar to those on leaves, sometimes girdling the plant if they occur near the soil line.')
                    st.write(
                        'Control:Early blight can be minimized by maintaining optimum growing conditions, including proper fertilization, irrigation, and management of other pests. Grow later maturing, longer season varieties. Fungicide application is justified only when the disease is initiated early enough to cause economic loss.')

            elif potato_disease == 'Common scab':
                with st.container():
                    p25 = Image.open('pics/2 potato/5.jpg')
                    st.image(p25, caption='fig. Common scab')

                    st.write(
                        'Symptoms : Pathogen infects young developing tubers through the lenticels and occasionally through wounds.Symptoms of common potato scab are quite variable and are manifested on the surface of the potato tuber. The disease forms several types of cork-like lesions including surface.')
                    st.write(
                        'Control : Use dolomitic lime after potatoes in the rotation. Applying sulphur to lower soil pH to between 5.0 and 5.2 can be useful in reducing the level of scab in some soils with high pH. Use acid-producing fertilisers and use ammonium sulphate as a source of nitrogen.')

            else:
                with st.container():
                    p26 = Image.open('pics/2 potato/6.jpeg')
                    st.image(p26, caption='fig. Black scurf/scurf')

                    st.write(
                        'Symptoms : Pathogen infects plant tissue and causes stolon blinding thus reducing tuber production and yield.It also infects tubers causing black scurf but this is purely cosmetic, reduces tuber appearance and does not reduce yield.')
                    st.write(
                        'Control : If there is a slight infection of black scurf that can be controlled by treating seed tubers with mercuric chloride solution for 1.5 hr with acidulated mercuric chloride solution for 5 min.')


        else:
            st.error('SELECT A CROP & DISEASE')

#COMPLETE COTTON DATABASE

    elif disease == 'COTTON':
        cotton_disease = st.selectbox('SELECT A COTTON DISEASE',options = {'Root rot',
                                                                    'Fusarium wilt',
                                                                    'Verticillium Wilt',
                                                                    'Para wilt',
                                                                    'Alterneria leaf blight',
                                                                    'Grey Mildew',
                                                                    'Anthracnose'})
        sub4 = st.button('SUBMIT')

        if sub4:
            if cotton_disease == 'Root rot':
                with st.container():
                    p31 = Image.open('pics/3 cotton/1.png')
                    st.image(p31, caption='fig. Root rot')

                    st.write(
                        'Symptoms : The root rot of cotton is caused by Rhizoctonia bataticola and Rhizoctonia solani fungi present in the soil supported by higher moisture in the soil.')
                    st.write(
                        'Control : Seed treatment with bio- agents Trichoderma viride 10 gm/kg + Pseudomonas fluorescens @ 10g/ kg seed or Thiram 75% WS 3g/kg seed or Soil drenching with Trichoderma viride @ 5 kg/acre mixed with 200 kg moist FYM. Spot drenching with Metalaxyl 35 % [Krilaxlyl Power] 1 g/L water at the base of affected plants as well as surrounding healthy plants.')

            elif cotton_disease == 'Fusarium wilt':
                with st.container():
                    p32 = Image.open('pics/3 cotton/2.jpeg')
                    st.image(p32, caption='fig. Fusarium wilt')

                    st.write(
                        'Symptoms : Fusarium wilt is common disease of cotton crop caused by the fungus Fusarium oxysporum, and capable of causing significant crop loss. The fungus is free living and may persist in soil as chlamydospores and in association with the roots of cotton as well as on the roots of weeds. Fusarium wilt spores may also live on the seeds of cotton.')
                    st.write(
                        'Control : Management of Fusarium wilt is difficult and most successfully achieved through the use of resistant cultivars and pathogen-free cotton seed. Once inoculum has been introduced into the field, strategies such as soil solarization and fumigation are applied to manage inoculum levels.')

            elif cotton_disease == 'Verticillium Wilt':
                with st.container():
                    p33 = Image.open('pics/3 cotton/3.jpeg')
                    st.image(p33, caption='fig. Verticillium Wilt')

                    st.write(
                        'Symptoms : Verticillium wilt of cotton disease is caused by Verticillium dahlia fungus. Reduced leaf size with mottling with yellow areas between veins and on margins; Brown necrotic leaves become dry and finally shed off.')
                    st.write(
                        'Control : How to Control Verticillium Wilt: There is no effective treatment for verticillium wilt. For affected vegetables, remove and dispose of the plant; dont compost it. For landscape plants, prune out affected branches and dispose of them immediately. Do not use infected wood for chips for landscape mulch.')

            elif cotton_disease == 'Para wilt':
                with st.container():
                    p34 = Image.open('pics/3 cotton/4.png')
                    st.image(p34, caption='fig. Para Wilt')

                    st.write(
                        'Symptoms : The sudden drying of cotton plants is noticed farmers’ fields after drought followed by rains or irrigation.')
                    st.write(
                        'Control : Seed treatment with bio- agents Trichoderma viride 10 gm/kg + Pseudomonas fluorescens @ 10g/ kg seed or Thiram 75% WS 3g/kg seed or Soil drenching with Trichoderma viride @ 5 kg/acre mixed with 200 kg moist FYM. Spot drenching with Metalaxyl 35 % [Krilaxlyl Power] 1 g/L water or Carbendazim 2 gm/L of water at the base of affected plants as well as surrounding healthy plants')

            elif cotton_disease == 'Alterneria leaf blight':
                with st.container():
                    p35 = Image.open('pics/3 cotton/5.jpg')
                    st.image(p35, caption='fig. Alterneria leaf blight')

                    st.write(
                        'Symptoms : Alternaria macrospora is the fungi which causes the disease. Brown irregular or round spots on leaves, and more spots merge to form larger spots. Sever infections leaves brittle and fall off.')
                    st.write(
                        'Control : Liquid copper fungicides such as Monterey Liqui-Cop or Bonide Liquid Copper Fungicide are extremely effective for the control of many fungal diseases including alternaria.')

            elif cotton_disease == 'Grey Mildew':
                with st.container():
                    p36 = Image.open('pics/3 cotton/6.jpg')
                    st.image(p36, caption='fig. Grey Mildew')

                    st.write(
                        'Symptoms : This disease is also called as Areolate mildew caused by Ramularia areola fungus. Translucent lesions on lower surface of the leaves with grey powdery mildew growth on the leaves and veins. Light green specks with white powdery mildew growth on upper side of the leaves.')
                    st.write(
                        'Control : Dusting of 8-10 kg of Sulphur powder effectively controls the disease. Also about one gram of Carbendazim or Benomyl per litre of water is effective. — If the disease intensity is more, new fungicides like one litre Hexaconazole or 300 gm Nativo-75 WG per hectare is required to control the grey mildew disease.')

            else:
                with st.container():
                    p37 = Image.open('pics/3 cotton/7.jpg')
                    st.image(p37, caption='fig. Anthracnose')

                    st.write(
                        'Symptoms : Anthracnose disease is one of the fungal disease which may affect all the plant parts of the cotton plant. At seedling stage, the circular red colored spots are noticed and seedlings may even die. Sunken, circular, reddish to brown colored spots on bolls and leaves are noticed at later stages of infection.')
                    st.write(
                        'Control : Treat the delinted seeds with Carbendazim or Carboxin or Thiram or Captan at 2g/kg. Remove and burn the infected plant debris and bolls in the soil. Rogue out the weed hosts. Spray the crop at boll formation stage with Mancozeb 2kg or Copper oxychloride 2.5 kg or or Carbendazim 500g/ha.')

        else:
            st.error('SELECT A CROP & DISEASE')

#COMPLETE PUMPKIN DATABASE

    elif disease == 'PUMPKIN':
        pumpkin_disease = st.selectbox('SELECT A PUMPKIN DISEASE',options = {'Alternaria leaf blight',
                                                                    'Cercospora leaf spot',
                                                                    'Downy mildew',
                                                                    'Fusarium crown and foot rot',
                                                                    'Gummy stem blight',
                                                                    'Southern blight'})
        sub5 = st.button('SUBMIT')

        if sub5:
            if pumpkin_disease == 'Alternaria leaf blight':
                with st.container():
                    p41 = Image.open('pics/4 pumpkin/1.jpg')
                    st.image(p41, caption='fig. Alternaria leaf blight')

                    st.write(
                        'Symptoms : Small, yellow-brown spots with a yellow or green halo which first appear on the oldest leaves; as the disease progresses, lesions expand and becone large necrotic patches, often with concentric patternation; lesions coalesce, leaves begin to curl and eventually die')
                    st.write(
                        'Control : Cucurbits should be rotated with another crop every 2 years to reduce levels of inoculum; crop debris should be removed from the field as quickly as possible after harvest or plowed deeply into the soil; applications of appropriate protective fungicides can help to slow the development of the disease; water plants from the base rather than from above to reduce periods of leaf wetness which are conducive to the development and spread of disease')


            elif pumpkin_disease == 'Cercospora leaf spot':
                with st.container():
                    p42 = Image.open('pics/4 pumpkin/2.jpeg')
                    st.image(p42, caption='fig. Cercospora leaf spot')

                    st.write(
                        'Symptoms : Initial symptoms of disease occur on older leaves as small spots with light to tan brown centers; as the disease progresses, the lesions enlarge to cover large areas of the leaf surface; lesions may have a dark border and be surrounded by a chlorotic area; the centers of the lesions may become brittle and crack')
                    st.write(
                        'Control : Any diseased plants should be removed and destroyed to prevent further spread; crop debris should be removed after harvest or plowed deeply into the soil to reduce inoculum')

            elif pumpkin_disease == 'Downy mildew':
                with st.container():
                    p43 = Image.open('pics/4 pumpkin/3.jpg')
                    st.image(p43, caption='fig. Downy mildew')

                    st.write(
                        'Symptoms : Dead or dying leaves; yellow to brown lesions on the upper side of leaves; purple growth developing on the underside of leaves')
                    st.write(
                        'Control : Do not overcrowd plants; avoid overhead irrigation, water plants from base; apply appropriate fungicide')

            elif pumpkin_disease == 'Fusarium crown and foot rot':
                with st.container():
                    p44 = Image.open('pics/4 pumpkin/4.jpeg')
                    st.image(p44, caption='fig. Fusarium crown and foot rot')

                    st.write(
                        'Symptoms : Wilting of leaves progresses to wilting of entire plant and plant dies within a few days; distinctive necrotic rot of crown and upper taproot when plant is uprooted; plant breaks easily below soil line.')
                    st.write(
                        'Control : Plant fungicide treated seed; rotate crops on 4 year rotation')

            elif pumpkin_disease == 'Gummy stem blight':
                with st.container():
                    p45 = Image.open('pics/4 pumpkin/5.jpg')
                    st.image(p45, caption='fig. Gummy stem blight')

                    st.write(
                        'Symptoms : Brown or tan spots of various sizes on leaves; leaves covered with lesions; stems splitting and forming cankers; wounds exude a brown, gummy substance; wilting vines; death of stems')
                    st.write(
                        'Control : Use disease free seed; treat seeds prior to planting; rotate crops every 2 years')

            else:
                with st.container():
                    p46 = Image.open('pics/4 pumpkin/6.jpeg')
                    st.image(p46, caption='fig. Southern blight')

                    st.write(
                        'Symptoms : Initial symptoms of disease are small dark water-soaked spots on the leaves which turn beige to white in dry conditions; lesions develop thin brown borders and the centers may become brittle and crack; small white spots may erupt on the surface of infected butternut and acorn squash and pumpkin fruit')
                    st.write(
                        'Control : Scout plants during cool wet conditions for any sign of spots; early application of an appropriate protective fungicide can help limit the development of the disease if spots are found, cucurbits should be rotated with other crops every 2 years to prevent the build-up of inoculum; crop debris should be removed and destroyed after harvest')

        else:
            st.error('SELECT A CROP & DISEASE')

#COMPLETE CABBAGE DATABASE

    elif disease == 'CABBAGE':
        cabbage_disease = st.selectbox('SELECT A CABBAGE DISEASE',options = {'Alternaria leaf blight',
                                                                    'Bacterial Leaf Spot',
                                                                    'Bacterial Soft Rot',
                                                                    'Black Rot',
                                                                    'Bottom Rot',
                                                                    'Clubroot',
                                                                    'Downy Mildew'})
        sub6 = st.button('SUBMIT')

        if sub6:
            if cabbage_disease == 'Alternaria leaf blight':
                with st.container():
                    p51 = Image.open('pics/5 cabbage/1.jpg')
                    st.image(p51, caption='fig. Alternaria leaf blight')

                    st.write(
                        'Symptoms : Symptoms include yellow spots that grow larger and develop rings around them like a target or bull’s-eye. As the tissue dies, the centers may fall out, resulting in holes in the foliage. As the disease develops, the spots join together to form large areas of dead tissue.')
                    st.write(
                        'Control : Controls for Alternaria leaf spot start with good cultural practices. These include the use of drip irrigation, sanitizing your gardening tools, rotating your crops, and removing all dead plant material at the end of the growing season.')

            elif cabbage_disease == 'Bacterial Leaf Spot':
                with st.container():
                    p52 = Image.open('pics/5 cabbage/2.jpg')
                    st.image(p52, caption='fig. Bacterial Leaf Spot')

                    st.write(
                        'Symptoms : Initial symptoms include dark flecks on the leaves that spread into lesions. The centers often degrade with time, resulting in circular holes in the foliage.')
                    st.write(
                        'Control : Prevention and control efforts range from planting heat-treated seed to irrigation practices that minimize the amount of moisture that comes in contact with the foliage. Crop rotation is also critical in subsequent seasons, as the bacteria can overwinter in the soil.')

            elif cabbage_disease == 'Bacterial Soft Rot':
                with st.container():
                    p53 = Image.open('pics/5 cabbage/3.jpg')
                    st.image(p53, caption='fig. Bacterial Soft Rot')

                    st.write(
                        'Symptoms : Cabbages can show symptoms in the field, but the majority of infections occur during storage. The first signs of infection are small lesions that appear water soaked. They quickly enlarge, and infected plant tissue turns brown and mushy.')
                    st.write(
                        'Control : There are a number of steps you can take to minimize the chances of your cabbages being infected, ranging from avoiding harvesting crops in wet conditions to removing any soil with a dry cloth before you store the heads.')

            elif cabbage_disease == 'Black Rot':
                with st.container():
                    p54 = Image.open('pics/5 cabbage/4.jpg')
                    st.image(p54, caption='fig.  Black Rot')

                    st.write(
                        'Symptoms : The first symptoms involve yellowing of the leaf margins, which then spread to the center of the leaf. A classic symptom is a yellow “V” at the midrib of the leaf. Next, the vascular system turns black, and the infection then spreads throughout the whole plant.')
                    st.write(
                        'Control : Planting high-quality seed that does not contain X. campestris pv. campestris is critical, and crop rotation will help to protect against infection. You also have the option of planting varieties of cabbage that are resistant. These include ‘Bobcat,’ ‘Guardian,’ and ‘Defender.’')

            elif cabbage_disease == ' Botttom Rot':
                with st.container():
                    p55 = Image.open('pics/5 cabbage/5.jpg')
                    st.image(p55, caption='fig.  Bottom Rot')

                    st.write(
                        'Symptoms : The initial symptoms are tan or brown lesions on the outer leaves. Then the fungus invades the center of the head, which can rot completely within 10 days.')
                    st.write(
                        'Control : There are no controls available once an infection has started.')

            elif cabbage_disease == 'Clubroot':
                with st.container():
                    p56 = Image.open('pics/5 cabbage/6.jpg')
                    st.image(p56, caption='fig.  Clubroot')

                    st.write(
                        'Symptoms : The initial symptoms are tan or brown lesions on the outer leaves. Then the fungus invades the center of the head, which can rot completely within 10 days.')
                    st.write(
                        'Control : There are no controls available once an infection has started.')

            else:
                with st.container():
                    p57 = Image.open('pics/5 cabbage/7.jpg')
                    st.image(p57, caption='fig.  Bottom Rot')

                    st.write(
                        'Symptoms : This pernicious and long-lived disease can be hard to detect. Older plants that are infected will wilt on hot days, but they can often appear to recover after the sun goes down. The pathogen enters the root hairs and then forms large club-like galls that can be as large as five or six inches wide. The roots don’t function properly, and are also left vulnerable to infection by other soil-borne pathogens.')
                    st.write(
                        'Control : Once a crop becomes infected, there are no effective control methods available. Since the most common source of infection is infected transplants, you should take care to use clean trays and seed starting media.')

        else:

            st.error('SELECT A CROP & DIEASE.')

