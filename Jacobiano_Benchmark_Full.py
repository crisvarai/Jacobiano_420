# ---------------------- Información del Script --------------------------

# PROGRAMA: Que efectúa el benchmark para el Jacobiano Analítico
# OBJETIVO: Anilizar el tiempo de cálculo que tomará calcular la matriz Jacobiana
# FECHA:    15 de Julio 2021
# DISEÑO:   Ing. Cristian Vallejo

import numpy as np
import timeit                               # For timing the durations

#------------------------- Declaración de parámetros DH ----------------------------#

q1 = ( 0 )*np.pi/180
q2 = ( -115 )*np.pi/180
q3 = ( -90 )*np.pi/180
q4 = ( 0 )*np.pi/180
q5 = ( 90 )*np.pi/180
q6 = ( 0 )*np.pi/180

d1 = 183.73 #dist. del pecho al hombro (q2)
a2 = 58.28
d3 = 244.4
a3 = 18.76
a4 = 18.53
d5 = 123.03
d7 = 150 #150 mm is dist. from q6 to ee; 71 mm, from q7.

k = 65*np.pi/180; 

def Jacobiano(q1,q2,q3,q4,q5,q6,a2,d3,a3,a4,d5,d7,k):

    #------------- Varianles sigma para reducir la expresión del Jacobiano -------------#

    sigma_66 = np.sin(k)*np.sin(q2) + np.cos(k)*np.cos(q1)*np.cos(q2)
    sigma_65 = np.cos(k)*np.sin(q2) - np.cos(q1)*np.cos(q2)*np.sin(k)
    sigma_64 = np.sin(q1)*np.sin(q3) - np.cos(q1)*np.cos(q2)*np.cos(q3)
    sigma_63 = np.cos(k)*np.cos(q1)*np.sin(q3) + np.cos(k)*np.cos(q2)*np.cos(q3)*np.sin(q1)
    sigma_62 = np.cos(q1)*np.sin(k)*np.sin(q3) + np.cos(q2)*np.cos(q3)*np.sin(k)*np.sin(q1)
    sigma_61 = np.cos(q2)*np.sin(k) - np.cos(k)*np.cos(q1)*np.sin(q2)
    sigma_60 = np.cos(k)*np.cos(q2) + np.cos(q1)*np.sin(k)*np.sin(q2)
    sigma_59 = np.cos(q1)*np.sin(q3) + np.cos(q2)*np.cos(q3)*np.sin(q1)
    sigma_58 = np.cos(q3)*sigma_66 - np.cos(k)*np.sin(q1)*np.sin(q3)
    sigma_57 = np.cos(q3)*sigma_65 + np.sin(k)*np.sin(q1)*np.sin(q3)
    sigma_56 = np.sin(q3)*sigma_65 - np.cos(q3)*np.sin(k)*np.sin(q1)
    sigma_55 = np.sin(q3)*sigma_66 + np.cos(k)*np.cos(q3)*np.sin(q1)
    sigma_54 = np.cos(q1)*np.cos(q3) - np.cos(q2)*np.sin(q1)*np.sin(q3)
    sigma_53 = np.cos(q3)*np.sin(q1) + np.cos(q1)*np.cos(q2)*np.sin(q3)
    sigma_52 = np.cos(q4)*sigma_64 + np.cos(q1)*np.sin(q2)*np.sin(q4)
    sigma_51 = np.cos(k)*np.cos(q1)*np.cos(q3) - np.cos(k)*np.cos(q2)*np.sin(q1)*np.sin(q3)
    sigma_50 = np.cos(q4)*sigma_63 - np.cos(k)*np.sin(q1)*np.sin(q2)*np.sin(q4)
    sigma_49 = np.cos(q1)*np.cos(q3)*np.sin(k) - np.cos(q2)*np.sin(k)*np.sin(q1)*np.sin(q3)
    sigma_48 = np.cos(q4)*sigma_62 - np.sin(k)*np.sin(q1)*np.sin(q2)*np.sin(q4)
    sigma_47 = np.sin(q4)*sigma_66 - np.cos(q3)*np.cos(q4)*sigma_61
    sigma_46 = np.sin(q4)*sigma_65 - np.cos(q3)*np.cos(q4)*sigma_60
    sigma_45 = np.cos(q4)*sigma_59 - np.sin(q1)*np.sin(q2)*np.sin(q4)
    sigma_44 = np.cos(q4)*sigma_58 + np.sin(q4)*sigma_61
    sigma_43 = np.cos(q4)*sigma_57 + np.sin(q4)*sigma_60
    sigma_42 = np.sin(q4)*sigma_57 - np.cos(q4)*sigma_60
    sigma_41 = np.sin(q4)*sigma_58 - np.cos(q4)*sigma_61
    sigma_40 = np.sin(q4)*sigma_59 + np.cos(q4)*np.sin(q1)*np.sin(q2)
    sigma_39 = np.sin(q5)*sigma_57 + np.cos(q4)*np.cos(q5)*sigma_56
    sigma_38 = np.sin(q5)*sigma_58 + np.cos(q4)*np.cos(q5)*sigma_55
    sigma_37 = np.cos(q2)*np.sin(q1)*np.sin(q4) + np.cos(q3)*np.cos(q4)*np.sin(q1)*np.sin(q2)
    sigma_36 = np.sin(q5)*sigma_59 - np.cos(q4)*np.cos(q5)*sigma_54
    sigma_35 = np.cos(q5)*sigma_52 + np.sin(q5)*sigma_53
    sigma_34 = np.sin(q4)*sigma_64 - np.cos(q1)*np.cos(q4)*np.sin(q2)
    sigma_33 = np.sin(q5)*sigma_51 + np.cos(q5)*sigma_50
    sigma_32 = np.sin(q4)*sigma_63 + np.cos(k)*np.cos(q4)*np.sin(q1)*np.sin(q2)
    sigma_31 = np.sin(q5)*sigma_49 + np.cos(q5)*sigma_48
    sigma_30 = np.sin(q4)*sigma_62 + np.cos(q4)*np.sin(k)*np.sin(q1)*np.sin(q2)
    sigma_29 = np.cos(q5)*sigma_47 + np.sin(q3)*np.sin(q5)*sigma_61
    sigma_28 = np.cos(q4)*sigma_66 + np.cos(q3)*np.sin(q4)*sigma_61
    sigma_27 = np.cos(q5)*sigma_46 + np.sin(q3)*np.sin(q5)*sigma_60
    sigma_26 = np.cos(q4)*sigma_65 + np.cos(q3)*np.sin(q4)*sigma_60
    sigma_25 = np.cos(q5)*sigma_45 + np.sin(q5)*sigma_54
    sigma_24 = np.sin(q5)*sigma_55 - np.cos(q5)*sigma_44
    sigma_23 = np.sin(q5)*sigma_56 - np.cos(q5)*sigma_43
    sigma_22 = np.cos(q6)*sigma_43 - np.cos(q5)*np.sin(q6)*sigma_42
    sigma_21 = np.cos(q6)*sigma_44 - np.cos(q5)*np.sin(q6)*sigma_41
    sigma_20 = np.cos(q6)*sigma_45 - np.cos(q5)*np.sin(q6)*sigma_40
    sigma_19 = np.sin(q6)*sigma_39 + np.cos(q6)*np.sin(q4)*sigma_56
    sigma_18 = np.sin(q6)*sigma_38 + np.cos(q6)*np.sin(q4)*sigma_55
    sigma_17 = np.cos(q5)*sigma_37 - np.sin(q1)*np.sin(q2)*np.sin(q3)*np.sin(q5)
    sigma_16 = np.sin(q6)*sigma_36 - np.cos(q6)*np.sin(q4)*sigma_54
    sigma_15 = np.cos(q6)*sigma_34 + np.sin(q6)*sigma_35
    sigma_14 = np.sin(q6)*sigma_33 + np.cos(q6)*sigma_32
    sigma_13 = np.cos(q6)*sigma_30 + np.sin(q6)*sigma_31
    sigma_12 = np.cos(q6)*sigma_28 - np.sin(q6)*sigma_29
    sigma_11 = np.cos(q6)*sigma_26 - np.sin(q6)*sigma_27
    sigma_10 = np.sin(q5)*sigma_45 - np.cos(q5)*sigma_54
    sigma_9  = np.cos(q6)*sigma_40 + np.sin(q6)*sigma_25
    sigma_8  = np.cos(q5)*sigma_55 + np.sin(q5)*sigma_44
    sigma_7  = np.cos(q5)*sigma_56 + np.sin(q5)*sigma_43
    sigma_6  = np.sin(q6)*sigma_40 - np.cos(q6)*sigma_25
    sigma_5  = np.sin(q6)*sigma_24 - np.cos(q6)*sigma_41
    sigma_4  = np.sin(q6)*sigma_23 - np.cos(q6)*sigma_42
    sigma_3  = np.sin(q6)*sigma_41 + np.cos(q6)*sigma_24
    sigma_2  = np.cos(q6)*sigma_23 + np.sin(q6)*sigma_42
    sigma_1  = np.cos(q2)*np.cos(q4)*np.sin(q1) - np.cos(q3)*np.sin(q1)*np.sin(q2)*np.sin(q4)

    #------------------- Ecuaciones para indexar en la matriz del Jacobiano --------------------#

    # Sección que representa la velocindad lineal

    Jv_11 = a4*np.cos(k)*np.sin(q1)*np.sin(q2)*np.sin(q4) - d7*sigma_14 - a4*np.cos(q4)*sigma_63 - a2*np.cos(k)*np.cos(q2)*np.sin(q1) - a3*np.cos(k)*np.cos(q1)*np.sin(q3) - d3*np.cos(k)*np.sin(q1)*np.sin(q2) - a3*np.cos(k)*np.cos(q2)*np.cos(q3)*np.sin(q1) - d5*sigma_32
    Jv_21 = a2*np.cos(q1)*np.cos(q2) - d7*sigma_15 - d5*sigma_34 + d3*np.cos(q1)*np.sin(q2) - a3*np.sin(q1)*np.sin(q3) - a4*np.cos(q4)*sigma_64 + a3*np.cos(q1)*np.cos(q2)*np.cos(q3) - a4*np.cos(q1)*np.sin(q2)*np.sin(q4)
    Jv_31 = a4*np.sin(k)*np.sin(q1)*np.sin(q2)*np.sin(q4) - d7*sigma_13 - a4*np.cos(q4)*sigma_62 - a2*np.cos(q2)*np.sin(k)*np.sin(q1) - a3*np.cos(q1)*np.sin(k)*np.sin(q3) - d3*np.sin(k)*np.sin(q1)*np.sin(q2) - a3*np.cos(q2)*np.cos(q3)*np.sin(k)*np.sin(q1) - d5*sigma_30

    Jv_12 = d3*sigma_66 + d5*sigma_28 + d7*sigma_12 + a2*np.cos(q2)*np.sin(k) + a3*np.cos(q3)*sigma_61 - a4*np.sin(q4)*sigma_66 - a2*np.cos(k)*np.cos(q1)*np.sin(q2) + a4*np.cos(q3)*np.cos(q4)*sigma_61
    Jv_22 = -np.sin(q1)*(a2*np.sin(q2) - d3*np.cos(q2) - d5*np.cos(q2)*np.cos(q4) + a3*np.cos(q3)*np.sin(q2) + a4*np.cos(q2)*np.sin(q4) - d7*np.cos(q2)*np.cos(q4)*np.cos(q6) + a4*np.cos(q3)*np.cos(q4)*np.sin(q2) + d5*np.cos(q3)*np.sin(q2)*np.sin(q4) + d7*np.cos(q3)*np.cos(q6)*np.sin(q2)*np.sin(q4) + d7*np.cos(q2)*np.cos(q5)*np.sin(q4)*np.sin(q6) - d7*np.sin(q2)*np.sin(q3)*np.sin(q5)*np.sin(q6) + d7*np.cos(q3)*np.cos(q4)*np.cos(q5)*np.sin(q2)*np.sin(q6))
    Jv_32 = a4*np.sin(q4)*sigma_65 - d5*sigma_26 - d7*sigma_11 - a2*np.cos(k)*np.cos(q2) - a3*np.cos(q3)*sigma_60 - d3*sigma_65 - a2*np.cos(q1)*np.sin(k)*np.sin(q2) - a4*np.cos(q3)*np.cos(q4)*sigma_60

    Jv_13 = -d7*sigma_18 - a4*np.cos(q4)*sigma_55 - d5*np.sin(q4)*sigma_55 - a3*np.sin(q3)*sigma_66 - a3*np.cos(k)*np.cos(q3)*np.sin(q1)
    Jv_23 = d5*np.sin(q4)*sigma_54 - d7*sigma_16 + a3*np.cos(q1)*np.cos(q3) + a4*np.cos(q4)*sigma_54 - a3*np.cos(q2)*np.sin(q1)*np.sin(q3)
    Jv_33 = d7*sigma_19 + a4*np.cos(q4)*sigma_56 + d5*np.sin(q4)*sigma_56 + a3*np.sin(q3)*sigma_65 - a3*np.cos(q3)*np.sin(k)*np.sin(q1)

    Jv_14 = d7*sigma_21 + d5*sigma_44 - a4*np.sin(q4)*sigma_58 + a4*np.cos(q4)*sigma_61
    Jv_24 = d7*sigma_20 + d5*sigma_45 - a4*np.sin(q4)*sigma_59 - a4*np.cos(q4)*np.sin(q1)*np.sin(q2)
    Jv_34 = a4*np.sin(q4)*sigma_57 - d5*sigma_43 - d7*sigma_22 - a4*np.cos(q4)*sigma_60

    Jv_15 = -d7*np.sin(q6)*sigma_8
    Jv_25 = -d7*np.sin(q6)*sigma_10
    Jv_35 = d7*np.sin(q6)*sigma_7

    Jv_16 = -d7*sigma_3
    Jv_26 = -d7*sigma_6
    Jv_36 = d7*sigma_2

    # Sección que representa la velocindad angular

    Jw_41 = sigma_10*(np.cos(q5)*sigma_49 - np.sin(q5)*sigma_48) - sigma_13*sigma_9 + (np.cos(q6)*sigma_31 - np.sin(q6)*sigma_30)*sigma_6
    Jw_51 = -(np.cos(q5)*sigma_51 - np.sin(q5)*sigma_50)*sigma_7 - sigma_14*sigma_4 - (np.cos(q6)*sigma_33 - np.sin(q6)*sigma_32)*sigma_2
    Jw_61 = sigma_5*sigma_15 - sigma_3*(np.sin(q6)*sigma_34 - np.cos(q6)*sigma_35) - (np.sin(q5)*sigma_52 - np.cos(q5)*sigma_53)*sigma_8

    Jw_42 = sigma_10*(np.sin(q5)*sigma_46 - np.cos(q5)*np.sin(q3)*sigma_60) - sigma_9*sigma_11 - sigma_6*(np.sin(q6)*sigma_26 + np.cos(q6)*sigma_27)
    Jw_52 = (np.sin(q5)*sigma_47 - np.cos(q5)*np.sin(q3)*sigma_61)*sigma_7 - (np.sin(q6)*sigma_28 + np.cos(q6)*sigma_29)*sigma_2 + sigma_12*sigma_4
    Jw_62 = sigma_3*(np.sin(q6)*sigma_1 + np.cos(q6)*sigma_17) - sigma_5*(np.cos(q6)*sigma_1 - np.sin(q6)*sigma_17) - (np.sin(q5)*sigma_37 + np.cos(q5)*np.sin(q1)*np.sin(q2)*np.sin(q3))*sigma_8

    Jw_43 = sigma_9*sigma_19 - sigma_10*(np.cos(q5)*sigma_57 - np.cos(q4)*np.sin(q5)*sigma_56) - sigma_6*(np.cos(q6)*sigma_39 - np.sin(q4)*np.sin(q6)*sigma_56)
    Jw_53 = -sigma_18*sigma_4 - (np.cos(q6)*sigma_38 - np.sin(q4)*np.sin(q6)*sigma_55*sigma_2) - (np.cos(q5)*sigma_58 - np.cos(q4)*np.sin(q5)*sigma_55)*sigma_7
    Jw_63 = (np.cos(q5)*sigma_59 + np.cos(q4)*np.sin(q5)*sigma_54)*sigma_8 + (np.cos(q6)*sigma_36 + np.sin(q4)*np.sin(q6)*sigma_54)*sigma_3 + sigma_16*sigma_5

    Jw_44 = np.sin(q5)*sigma_42*sigma_10 - sigma_6*(np.sin(q6)*sigma_43 + np.cos(q5)*np.cos(q6)*sigma_42) - sigma_9*sigma_22
    Jw_54 = sigma_21*sigma_4 - (np.sin(q6)*sigma_44 + np.cos(q5)*np.cos(q6)*sigma_41)*sigma_2 + np.sin(q5)*sigma_41*sigma_7
    Jw_64 = (np.sin(q6)*sigma_45 + np.cos(q5)*np.cos(q6)*sigma_40)*sigma_3 - sigma_20*sigma_5 - np.sin(q5)*sigma_40*sigma_8

    Jw_45 = sigma_10*sigma_23 - np.cos(q6)*sigma_6*sigma_7 + np.sin(q6)*sigma_9*sigma_7
    Jw_55 = sigma_24*sigma_7 - np.cos(q6)*sigma_8*sigma_2 - np.sin(q6)*sigma_8*sigma_4
    Jw_65 = sigma_25*sigma_8 + np.cos(q6)*sigma_10*sigma_3 + np.sin(q6)*sigma_10*sigma_5

    Jw_46 = sigma_9*sigma_2 + sigma_6*sigma_4
    Jw_56 = sigma_2*sigma_5 - sigma_3*sigma_4
    Jw_66 = sigma_9*sigma_3 + sigma_6*sigma_5

    np.set_printoptions(precision=4, suppress=True)

    J = np.array([[Jv_11,Jv_12,Jv_13,Jv_14,Jv_15,Jv_16], 
                [Jv_21,Jv_22,Jv_23,Jv_24,Jv_25,Jv_26], 
                [Jv_31,Jv_32,Jv_33,Jv_34,Jv_35,Jv_36],
                [Jw_41,Jw_42,Jw_43,Jw_44,Jw_45,Jw_46],
                [Jw_51,Jw_52,Jw_53,Jw_54,Jw_55,Jw_56],
                [Jw_61,Jw_62,Jw_63,Jw_64,Jw_65,Jw_66]])

    return J

tStart = timeit.default_timer()                             # Record our starting time
passes = 0                                                  # We're going to count how many passes we make in fixed window of time

while (timeit.default_timer() - tStart < 1):               # Run until more than 1 seconds have elapsed
    J = Jacobiano(q1,q2,q3,q4,q5,q6,a2,d3,a3,a4,d5,d7,k)    # Find the results
    passes = passes + 1                                     # Count this pass

tD = timeit.default_timer() - tStart                        # After the "at least 1 seconds", get the actual elapsed
print("tD, Passes = {}, {}".format(tD, passes))
