# archivo de estilos de gráficos
import palettable
import matplotlib.pyplot as plt

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palettable.scientific.sequential.Oslo_9.mpl_colors)
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['font.family'] = 'times new roman'

plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.titleweight'] = 'normal'
# plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['legend.fontsize'] = 10

plt.rcParams['date.autoformatter.day'] = '%d-%m-%Y'
plt.rcParams['date.autoformatter.hour'] = '%H %d-%m'
plt.rcParams['date.autoformatter.minute'] = '%H:%M %d'
plt.rcParams['date.autoformatter.month'] = '%m-%Y'

# Temporary styling:
# with plt.style.context('dark_background'):
#     plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
# plt.show()