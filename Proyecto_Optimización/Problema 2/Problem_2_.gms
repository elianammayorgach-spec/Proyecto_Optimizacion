$offSymXRef
$offSymList

option limrow = 0;
option limcol = 0;
option solprint = on;
option sysout = off;

option LP   = CPLEX;
option MIP  = CPLEX;
option NLP  = CONOPT;
option MINLP= DICOPT;
option OPTCR= 0;

$title Problem 2 - HEN Optimization (Biodiesel 10,000 ton/y)

$onText

Problem 2: Process Optimization Project           
Red de Intercambiadores de Calor - Planta de Biodiesel
Professor: Francisco Javier Vasquez Vasquez       
EMMCH

$offText

Sets
    h   Hot streams       /H1*H4/
    c   Cold streams      /C1*C4/ ;

Parameters
    Tin_h(h)        Temperatura inicial caliente (K)
    Tout_h(h)       Temperatura objetivo caliente (K)
    mcp_h(h)        Capacidad calorífica caliente (kW*K^-1)
    Tin_c(c)        Temperatura inicial  fría (K)
    Tout_c(c)       Temperatura objetivo fría (K)
    mcp_c(c)        Capacidad calorífica fría (kW*K^-1)
    DTmin           Delta T mínima (K)
    U               Coeficiente global de transferencia (kW*m2*K^-1)))
    alpha           Factor de anualización
    hours           Horas de operación por año
    costSteam_kWh   Costo térmico del vapor (USD*kWh^-1)
    costCW_kWh      Costo térmico del agua (USD*kWh^-1)
    MQ(h,c)         Big-M para calor (kW)
    MA(h,c)         Big-M para área (m2)
    MDelta          Big-M factibilidad térmica (K)
    TotalColdLoad   Carga total fría (kW) ;

DTmin   = 10;
U       = 800;
alpha   = 0.15;
hours   = 8000;
MDelta  = 1000;

* Costos térmicos
costSteam_kWh = 25 / 596;     
costCW_kWh    = 0.05 / 11.63;

* Datos de corrientes (biodiesel)
Tin_h('H1')  = 448.15;     Tout_h('H1') = 363.15;    mcp_h('H1') = 2.309495;
Tin_h('H2')  = 363.15;     Tout_h('H2') = 323.15;    mcp_h('H2') = 11.925;
Tin_h('H3')  = 391.15;     Tout_h('H3') = 323.15;    mcp_h('H3') = 3.764706;
Tin_h('H4')  = 493.15;     Tout_h('H4') = 355.15;    mcp_h('H4') = 110.9203;

Tin_c('C1')  = 298.15;     Tout_c('C1') = 330.15;    mcp_c('C1') = 1.372881;
Tin_c('C2')  = 355.15;     Tout_c('C2') = 493.15;    mcp_c('C2') = 15.24638;
Tin_c('C3')  = 323.15;     Tout_c('C3') = 373.15;    mcp_c('C3') = 1.7;
Tin_c('C4')  = 330.15;     Tout_c('C4') = 400.15;    mcp_c('C4') = 221.5641;

* Big-M por disponibilidad máxima
MQ(h,c) = min( mcp_h(h)*(Tin_h(h)-Tout_h(h)), mcp_c(c)*(Tout_c(c)-Tin_c(c)) );
MA(h,c) = MQ(h,c) / (U*DTmin);

* Carga fría total
TotalColdLoad = sum(c, mcp_c(c)*(Tout_c(c)-Tin_c(c)));

* Cobertura mínima por corrientes frías (empuja intercambio en corrientes pequeñas)
Parameter gammaC(c)  Fracción mínima interna por corriente fría;
gammaC(c)      = 0;
gammaC('C1')   = 0.50;  
gammaC('C3')   = 0.90;
gammaC('C2')   = 0.00;  
gammaC('C4')   = 0.95; 

Variables
    Q(h,c)              Intercambio de calor (kW)
    A(h,c)              Área de transferencia (m2)
    z(h,c)              Match binario
    QSteam(c)           Calentamiento externo frío (kW)
    QCW(h)              Enfriamiento caliente (kW)
    Cost                Costo total (USD)*año^-1)
    Cost_CAPEX_Fijo     Componente CAPEX fijo (USD*año^-1)
    Cost_CAPEX_Area     Componente CAPEX por área (USD*año^-1)
    Cost_OPEX_Steam     Componente OPEX vapor (USD*año^-1)
    Cost_OPEX_CW        Componente OPEX agua (USD*año^-1)
    Qsum                Suma de intercambio interno (kW)
    CostStage1          Objetivo etapa 1 (proxy) ;

Positive Variables Q, A, QSteam, QCW;
Binary Variables z;

Parameter RHSHold;

Equations
    HeatBalance_H(h)
    HeatBalance_C(c)
    AreaDef(h,c)
    Link_Q(h,c)
    Link_A(h,c)
    NoEmptyMatch(h,c)
    FeasHotIn_ColdOut(h,c)
    FeasHotOut_ColdIn(h,c)
    DefCost_CAPEX_Fijo
    DefCost_CAPEX_Area
    DefCost_OPEX_Steam
    DefCost_OPEX_CW
    Objective
    MinRecPerCold(c)
    ObjMaxQ
    HoldInternal;

* Balances de energía
HeatBalance_H(h)..     sum(c, Q(h,c)) + QCW(h) =E= mcp_h(h)*(Tin_h(h)-Tout_h(h));
HeatBalance_C(c)..     sum(h, Q(h,c)) + QSteam(c) =E= mcp_c(c)*(Tout_c(c)-Tin_c(c));

* Área (con ΔTmin simplificado; conserva A**0.8 en costo)
AreaDef(h,c)..         A(h,c) =E= Q(h,c) / (U * DTmin);

* Enlaces binario–continuo
Link_Q(h,c)..          Q(h,c) =L= MQ(h,c) * z(h,c);
Link_A(h,c)..          A(h,c) =L= MA(h,c) * z(h,c);

* Evitar matches vacíos
NoEmptyMatch(h,c)..    Q(h,c) =G= 1e-3*z(h,c);

* Factibilidad térmica por extremos (contracorriente)
FeasHotIn_ColdOut(h,c).. Tin_h(h) - Tout_c(c) =G= DTmin - MDelta*(1 - z(h,c));
FeasHotOut_ColdIn(h,c).. Tout_h(h) - Tin_c(c) =G= DTmin - MDelta*(1 - z(h,c));

* Costos (manteniendo A**0.8)
DefCost_CAPEX_Fijo..   Cost_CAPEX_Fijo =E= sum((h,c), alpha*15000*z(h,c));
DefCost_CAPEX_Area..   Cost_CAPEX_Area =E= sum((h,c), alpha*500*(A(h,c)**0.8));
DefCost_OPEX_Steam..   Cost_OPEX_Steam =E= hours*costSteam_kWh*sum(c, QSteam(c));
DefCost_OPEX_CW..      Cost_OPEX_CW    =E= hours*costCW_kWh   *sum(h, QCW(h));

* Objetivo etapa 2 (costo total)
Objective..            Cost =E= Cost_CAPEX_Fijo + Cost_CAPEX_Area + Cost_OPEX_Steam + Cost_OPEX_CW;

* Cobertura mínima por corriente fría
MinRecPerCold(c)..     sum(h, Q(h,c)) =G= gammaC(c) * mcp_c(c) * (Tout_c(c) - Tin_c(c));

* Objetivo etapa 1: maximizar intercambio interno
ObjMaxQ..              Qsum =E= sum((h,c), Q(h,c));

Model MaxRec / HeatBalance_H, HeatBalance_C, AreaDef, Link_Q, Link_A, NoEmptyMatch,
              FeasHotIn_ColdOut, FeasHotOut_ColdIn, MinRecPerCold, ObjMaxQ /;


Model Problem2 / HeatBalance_H, HeatBalance_C, AreaDef, Link_Q, Link_A, NoEmptyMatch,
                 FeasHotIn_ColdOut, FeasHotOut_ColdIn, MinRecPerCold,
                 DefCost_CAPEX_Fijo, DefCost_CAPEX_Area, DefCost_OPEX_Steam, DefCost_OPEX_CW,
                 Objective, HoldInternal /;


* Anexos
Q.up(h,c)      = MQ(h,c);
A.up(h,c)      = MA(h,c);
QSteam.up(c)   = mcp_c(c)*(Tout_c(c)-Tin_c(c));
QCW.up(h)      = mcp_h(h)*(Tin_h(h)-Tout_h(h));

* Semillas para guiar el MINLP 
z.L('H4','C4') = 1;   Q.L('H4','C4') = 2000;
z.L('H3','C1') = 1;   Q.L('H3','C1') =   40;
z.L('H1','C3') = 1;   Q.L('H1','C3') =   60;

* ===== Etapa 1: Maximizar Q interno =====
Solve MaxRec using MINLP maximizing Qsum;

* Capturar el nivel de intercambio interno alcanzado
Scalar QTarget, holdFrac;
QTarget  = sum((h,c), Q.l(h,c));
holdFrac = 0.95; 

* Añadir restricción para etapa 2 (mantener intercambio interno elevado)
Equations HoldInternal;
HoldInternal..  sum((h,c), Q(h,c)) =G= RHSHold;

* ===== Etapa 1: Maximizar Q interno =====
Solve MaxRec using MINLP maximizing Qsum;

* Capturar el nivel logrado y fijar el RHS de la restricción Hold
Scalar QTarget, holdFrac;
QTarget  = sum((h,c), Q.l(h,c));
holdFrac = 0.98;  
RHSHold  = holdFrac * QTarget;

* ===== Etapa 2: Minimizar costo manteniendo intercambio interno =====
Solve Problem2 using MINLP minimizing Cost;

* Reporte
scalar modelstat1, solvestat1, modelstat2, solvestat2;
modelstat1 = MaxRec.modelstat;   solvestat1 = MaxRec.solvestat;
modelstat2 = Problem2.modelstat; solvestat2 = Problem2.solvestat;
display modelstat1, solvestat1, modelstat2, solvestat2;

display Qsum.l, QTarget, holdFrac, RHSHold;

display Cost.l, Cost_CAPEX_Fijo.l, Cost_CAPEX_Area.l, Cost_OPEX_Steam.l, Cost_OPEX_CW.l;

Scalar MatchesActive;
MatchesActive = sum((h,c), z.l(h,c));
display MatchesActive;

display z.l, Q.l, QSteam.l, QCW.l, A.l;