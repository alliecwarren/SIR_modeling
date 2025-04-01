from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64
import os
import sys
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.io as pio
from plotly.io import to_html
from datetime import datetime
from scipy.integrate import solve_ivp
from numpy.random import poisson

plt.style.use('ggplot')

# Sources:
# reference: https://medium.com/data-science/simulating-compartmental-models-in-epidemiology-using-python-jupyter-widgets-8d76bdaff5c2
# inspired by: https://epiengage-measles.tacc.utexas.edu/
# data from: https://doh.wa.gov/data-and-statistical-reports/washington-tracking-network-wtn/school-immunization/dashboard


# Vaccination Data
base_folder = "C:/Users/ACW3303/sir_models/data"
wa_vaccination = pd.read_csv(base_folder + "measles_Building Data Download_data_2023_2024.csv")
wa_schools = pd.read_csv(base_folder + "School Building Level_data.csv")
wa_school_vaccination = pd.merge(wa_vaccination, wa_schools, how = 'left', on = ['School Name', 'City', 'Grade'])
vacc_subset = wa_school_vaccination[wa_school_vaccination['Group'] == 'Complete']
vacc_subset = vacc_subset[vacc_subset['Measure Names'].isin(['Enrollment', "%"])]
vaccination_rates = vacc_subset[vacc_subset['Measure Names'] == '%']
school_enrollment = vacc_subset[vacc_subset['Measure Names'] == 'Enrollment']


def ode_model(z, t, beta, sigma, gamma, mu):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    Ordinary Differential Equation
    returns system of ODE in list
    """
    S, E, I, R, D = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def ode_solver(t, initial_conditions, params):
    """
    ODE Solver - takes in input time span, and initial conditions
    for solving system of ODE
    returns resulting prediction after solving 
    """
    initE, initI, initR, initN, initD = initial_conditions
    beta, sigma, gamma, mu = params
    initS = initN - (initE + initI + initR + initD)
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu))
    return res

def stochastic_seird_model(initial_conditions, t_span, t_step, 
                          beta_mean, sigma_mean, gamma_mean, mu_mean):
    """
    Stochastic SEIRD model with parameters drawn from Poisson distributions
    
    Args:
        initial_conditions: tuple (initS, initE, initI, initR, initD)
        t_span: tuple (t_start, t_end)
        t_step: time step for simulation
        beta_mean, sigma_mean, gamma_mean, mu_mean: mean values for Poisson distributions
    """
    # Initialize arrays to store results
    t_points = np.arange(t_span[0], t_span[1] + t_step, t_step)
    n_steps = len(t_points)
    results = np.zeros((5, n_steps))
    
    # Set initial conditions
    S, E, I, R, D = initial_conditions
    results[0,0] = S
    results[1,0] = E
    results[2,0] = I
    results[3,0] = R
    results[4,0] = D
    
    # Run simulation
    for i in range(1, n_steps):
        # Generate random parameters from Poisson distributions
        # Note: Poisson can only generate integers, so we scale appropriately
        scale_factor = 50  # Adjust based on your parameter magnitudes
        
        beta = poisson(beta_mean * scale_factor) / scale_factor
        sigma = poisson(sigma_mean * scale_factor) / scale_factor
        gamma = poisson(gamma_mean * scale_factor) / scale_factor
        mu = poisson(mu_mean * scale_factor) / scale_factor
        # Get current state
        S = results[0,i-1]
        E = results[1,i-1]
        I = results[2,i-1]
        R = results[3,i-1]
        D = results[4,i-1]        
        N = S + E + I + R + D
        
        # Calculate derivatives
        dSdt = -beta*S*I/N
        dEdt = beta*S*I/N - sigma*E
        dIdt = sigma*E - gamma*I - mu*I
        dRdt = gamma*I
        dDdt = mu*I
        
        # Update using Euler method
        S_new = S + dSdt * t_step
        E_new = E + dEdt * t_step
        I_new = I + dIdt * t_step
        R_new = R + dRdt * t_step
        D_new = D + dDdt * t_step

        # Ensure non-negative values
        S_new = max(0, S_new)
        E_new = max(0, E_new)
        I_new = max(0, I_new)
        R_new = max(0, R_new)
        D_new = max(0, D_new)
        
        # Store results
        results[0,i] = S_new
        results[1,i] = E_new
        results[2,i] = I_new
        results[3,i] = R_new
        results[4,i] = D_new
    
    return t_points, results

# Example usage:
def run_simulation(initial_conditions = (200, 1, 1, 0, 0), t_max =100, beta_mean=3, sigma_mean=(1/10.5), gamma_mean=(1/5), mu_mean=(1.5/1000)):
        t_span = (0, t_max)
        t_step = 0.1  # Time step
        
        t, results = stochastic_seird_model(
            initial_conditions, t_span, t_step, 
            beta_mean, sigma_mean, gamma_mean, mu_mean
        )
        
        return t, results

def calc_potential_infectious(num_scenarios, initial_conditions, t_max, beta_mean, sigma_mean, gamma_mean, mu_mean):
        t_span = (0, t_max)
        t_step = 0.1 
        t_points = np.arange(t_span[0], t_span[1] + t_step, t_step)
        n_steps = len(t_points)
        infectious_results = np.zeros((num_scenarios, n_steps))
        for i in range(num_scenarios):
            time_list, results = run_simulation(initial_conditions, t_max, beta_mean, sigma_mean, gamma_mean, mu_mean)
            infectious_results[i] = results[2]
        
        return time_list, infectious_results

def plot_scenarios(num_scenarios = 10, initial_conditions = (200, 1, 1, 0, 0), t_max =100, beta_mean=3, sigma_mean=(1/10.5), gamma_mean=(1/5), mu_mean=(1.5/1000)):
    time_list, infectious_results = calc_potential_infectious(num_scenarios, initial_conditions, t_max, beta_mean, sigma_mean, gamma_mean, mu_mean)
    # Create traces
    days = max(time_list)

    fig = go.Figure()
    for i in range(len(infectious_results)):
        fig.add_trace(go.Scatter(x=time_list, y=infectious_results[i,:], mode='lines', name='scenario ' + str(i)))
        
    if days <= 30:
        step = 1
    elif days <= 100:
        step = 10
    else:
        step = 25
    
    # Edit the layout
    fig.update_layout(title=str(len(infectious_results)) + ' Simulations of Infected Counts from SEIRD Model',
                    xaxis_title='Day',
                    yaxis_title='Number Infected',
                    title_x=0.5,
                    width=900, height=600
                    )
    fig.update_xaxes(tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
    
    return fig


# Assume we have this function that creates plots based on number of students
def create_plot(initN, initE = 1, initI = 1, initR = 0, initD = 0, R0 = 15, sigma = (1/10.5), gamma = (1/5), mu = (1.5/1000), days = 100):
    """
    Creates a plot of SEIRD model based on given parameters
    initN: population size (number of susceptible students)
    initE: initial exposed, default 1
    initI: initial infected, default 1
    initR: initial recovered, default 0
    initD: initial dead, default 0
    R0: basic reproduction number, default 15
    sigma: 1/average incubration/latent period (days), default 1/10.5 
    gamma: 1/average infection period (days), default 1/5 
    mu: number of deaths, default 1.5/1000
    days: number of days to simulate, default 100
    """
    beta = R0 * gamma

    initial_conditions = [initE, initI, initR, initN, initD]
    params = [beta, sigma, gamma, mu]
    tspan = np.arange(0, days, 1)
    sol = ode_solver(tspan, initial_conditions, params)
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines+markers', name='Susceptible'))
    fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines+markers', name='Exposed'))
    fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines+markers', name='Infected'))
    fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines+markers',name='Recovered'))
    fig.add_trace(go.Scatter(x=tspan, y=D, mode='lines+markers',name='Death'))
    
    # based on number of days being simulated adjust x-axis tick labels on plot
    if days < 30:
        step = 1
    elif days < 150:
        step = 10
    else:
        step = 25

    # Edit the layout
    fig.update_layout(title='Simulation of SEIRD Model',
                       xaxis_title='Day',
                       yaxis_title='Counts',
                       title_x=0.5,
                      width=900, height=600
                     )
    fig.update_xaxes(tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
    return fig

# Define the SEIRD simulations using school data tab
location_tab = ui.nav_panel(
    "Simulations Using School Parameters",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Select School"),
            ui.input_select("county", "County:", choices=sorted(vacc_subset['County'].unique())),
            ui.output_ui("district_ui"),
            ui.output_ui("school_ui"),
        ),
        ui.h3("School Information"),
        ui.output_text_verbatim("selected_school_info"),
        ui.output_ui("location_plot")
    )
)

# Define the SEIRD simulation using custom parameters tab
students_tab = ui.nav_panel(
    "Set Custom Parameters",
    ui.layout_sidebar(
        ui.sidebar(
            ui.p("Select parameters:"),
            ui.input_slider("num_students", "Total Number of Students:", 
                          min=10, 
                          max=10000,
                          value=500),
            ui.input_slider("vax_rate", "Vaccination Rate:", 
                          min=.01, max=.99, value=.8, step=0.01),
            ui.input_slider("R0", "R0 (Basic Reproduction Number):", 
                          min=1, max=20, value=15, step=0.5),
            ui.input_slider("sigma", "Average Latent Period (days):", 
                          min=5, max=20, value=10.5, step=0.5),
            ui.input_slider("gamma", "Average Infectious Period (days):", 
                          min=1, max=15, value=5, step=0.5),
            ui.input_slider("deaths", "Average Number of deaths per 1000 cases:", 
                          min=0.1, max=5, value=1.5, step=0.1),              
            ui.input_slider("days", "Simulation Days:", 
                          min=30, max=200, value=100, step=10),
        ),
        ui.h3("SEIRD Model Simulation"),
        ui.output_ui("student_plot") 
    )
)

# Define the SEIRD simulation using custom parameters tab
stochastic_tab = ui.nav_panel(
    "Stochastic Simulation of Infections",
    ui.layout_sidebar(
        ui.sidebar(
            ui.p("Select parameters:"),
            ui.input_slider("total_num", "Total Number of Students:", 
                          min=10, 
                          max=10000,
                          value=500),
            ui.input_slider("vaccine_rate", "Vaccination Rate:", 
                          min=.01, max=.99, value=.8, step=0.01),
            ui.input_slider("R_0", "R0 (Basic Reproduction Number):", 
                          min=1, max=20, value=15, step=0.5),
            ui.input_slider("sigma_2", "Average Latent Period (days):", 
                          min=5, max=20, value=10.5, step=0.5),
            ui.input_slider("gamma_2", "Average Infectious Period (days):", 
                          min=1, max=15, value=5, step=0.5),
            ui.input_slider("num_death", "Average Number of deaths per 1000 cases:", 
                          min=0.1, max=5, value=1.5, step=0.1),              
            ui.input_slider("num_days", "Simulation Days:", 
                          min=30, max=200, value=100, step=10),
            ui.input_slider("num_simulations", "Number of Simulations:", 
                          min=1, max=20, value=10, step=1),
        ),
        ui.h3("Number Infected Simulation"),
        ui.output_ui("infected_plot") 
    )
)

# Define the UI
app_ui = ui.page_fluid(
    ui.h1("Measles Outbreak Simulation in Schools"),
    ui.navset_tab(
        location_tab,
        students_tab,
        #stochastic_tab,
        id="tabs"
    )
)

# Define server logic
def server(input, output, session):
    
    # Dynamic UI elements for district selection based on county
    @output
    @render.ui
    def district_ui():
        county = input.county()
        districts = sorted(vacc_subset[vacc_subset['County'] == county]['School District'].unique())
        return ui.input_select("district", "School District:", choices=districts)
    
    # Dynamic UI elements for school selection based on district
    @output
    @render.ui
    def school_ui():
        # Check if district input exists and has a value
        if not hasattr(input, 'district') or input.district() is None:
            return ui.p("Select a district first")
        
        district = input.district()
        schools = sorted(vacc_subset[vacc_subset['School District'] == district]['School Name'].unique())
        return ui.input_select("school", "School:", choices=schools)
    
    # Reactive expression to get selected school information
    @reactive.Calc
    @reactive.event(input.county, input.district, input.school)
    def get_selected_school_info():
        # Check if school input exists and has a value
        if not hasattr(input, 'school') or input.school() is None:
            return None
        
        school = input.school()
        school_data = vacc_subset[vacc_subset['School Name'] == school]
        if len(school_data) == 0:
            return None
        return school_data
    
    # Display selected school information
    @output
    @render.text
    def selected_school_info():
        school_data = get_selected_school_info()
        if school_data is None:
            return "Please select a school"
        
        # based on school enrollment numbers and vaccination rate calculate the number of
        # susceptible students, assumes that .0005 vaccinated individuals are still susceptible,
        # otherwise assumes complete protection from vaccination. Assumes unvaccinated individuals
        # have no protection
        totalN = school_data[school_data['Measure Names'] == 'Enrollment'].iloc[0]['Measure Values']
        vaccination_rate = school_data[school_data['Measure Names'] == '%'].iloc[0]['Measure Values']
        vacc_failure_rate = 0.0005
        no_vacc = totalN * (1-vaccination_rate)
        susceptible_vacc = (totalN * vaccination_rate) * vacc_failure_rate
        initN = np.ceil(no_vacc + susceptible_vacc)
        # display school parameters
        return (f"County: {school_data.iloc[0]['County']}\n"
                f"School District: {school_data.iloc[0]['School District']}\n"
                f"School: {school_data.iloc[0]['School Name']}\n"
                f"Number of Students: {totalN}\n"
                f"Vaccination Rate: {vaccination_rate}\n"
                f"Number of Susceptible Students: {initN}")
       
    
    # Plot based on school selection
    @output
    @render.ui
    def location_plot():
        school_data = get_selected_school_info()
        timestamp = datetime.now().isoformat()
        # create default plot if no data selected
        if school_data is None:
            fig = create_plot(initN=100)
            plot_html = to_html(fig, include_plotlyjs="cdn", full_html=True)
        else:
            # calculate number of susceptible students given school data, otherwise use default parameters
            totalN = school_data[school_data['Measure Names'] == 'Enrollment'].iloc[0]['Measure Values']
            vaccination_rate = school_data[school_data['Measure Names'] == '%'].iloc[0]['Measure Values']
            vacc_failure_rate = 0.0005
            no_vacc = totalN * (1-vaccination_rate)
            susceptible_vacc = (totalN * vaccination_rate) * vacc_failure_rate
            initS = np.ceil(no_vacc + susceptible_vacc)
            vaccine_protected = totalN - initS

            fig = create_plot(initN=totalN, initR = vaccine_protected)
            plot_html = to_html(fig, include_plotlyjs="cdn", full_html=True)
        #  Add a comment with timestamp to prevent caching
        plot_html += f"<!-- {timestamp} -->"
        return ui.HTML(plot_html)
    
    
    # Plot based on selected parameters
    @output
    @render.ui
    def student_plot():
        num_students = input.num_students()
        r0 = input.R0()
        vaccination_rate = input.vax_rate()
        days = input.days()
        sigma_value = input.sigma()
        gamma_value = input.gamma()
        num_deaths = input.deaths()
        # calculate the number of susceptible students given the selected parameters
        vacc_failure_rate = 0.0005
        no_vacc = num_students * (1-vaccination_rate)
        susceptible_vacc = (num_students * vaccination_rate) * vacc_failure_rate
        initS = np.ceil(no_vacc + susceptible_vacc)
        vaccine_protected = num_students - initS

        fig = create_plot(initN=num_students, R0=r0, initR = vaccine_protected, days=days, sigma = (1/sigma_value), gamma = (1/gamma_value), mu = (num_deaths/1000),)
        plot_html = to_html(fig, include_plotlyjs="cdn", full_html=False)
        return ui.HTML(plot_html)
    
    # Plot based on selected parameters
    @output
    @render.ui
    def infected_plot():
        num_students = input.total_num()
        r0 = input.R_0()
        vaccination_rate = input.vaccine_rate()
        days = input.num_days()
        sigma_value = input.sigma_2()
        gamma_value = input.gamma_2()
        num_deaths = input.num_death()
        simulations = input.num_simulations()
        # calculate the number of susceptible students given the selected parameters
        vacc_failure_rate = 0.0005
        no_vacc = num_students * (1-vaccination_rate)
        susceptible_vacc = (num_students * vaccination_rate) * vacc_failure_rate
        initS = np.ceil(no_vacc + susceptible_vacc)
        vaccine_protected = num_students - initS
        beta_value = r0 * (1/gamma_value)
        fig = plot_scenarios(num_scenarios=simulations, initial_conditions=(initS, 1, 1, vaccine_protected, 0), t_max = days, beta_mean=beta_value,
                             sigma_mean=(1/sigma_value), gamma_mean=(1/gamma_value), mu_mean=(num_deaths/1000))
        plot_html = to_html(fig, include_plotlyjs="cdn", full_html=False)
        return ui.HTML(plot_html)
    
# Create and run the app
app = App(app_ui, server)

# Run the app
if __name__ == "__main__":
    app.run()