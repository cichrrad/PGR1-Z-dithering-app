# DitherApp Readme

## OBSAH

### [1] Obecné informace
### [2] Uživatelské rozhraní
### [3] Závislosti (dependencies)

## [1] OBECNÉ INFORMACE

Program DitherApp se skládá ze 4 .py souborů:

- **DitherApp.py**: Hlavní soubor (**PRO ZAPNUTÍ APLIKACE JE TŘEBA RUNNOUT TENTO SOUBOR**) obsahující třídu root - aplikace je instance této třídy. Také obsahuje funkce vykonávající základní operace (načtení/uložení) a algoritmy pro úpravu obrazu. 
 
- **ImageWidgets.py**: Soubor obsahující prvky GUI vztahující se k prvotnímu načtení obrázku a GUI prvek na kterém se vykresluje obrázek (Canvas).

- **ControlWidgets.py**: Soubor obsahující prvky GUI vztahující se k ovládání programu (levý ovládací sloupec s tlačítky). Logika/funkce přidávající prvkům funkcionalitu jsou ale v DitherApp.py.

- **matrixDefinitions.py**: Soubor obsahující různé matice pro Error diffusion.

Program Uživateli dovoluje načíst obrázek (ve standardních formátech) a aplikovat na něj jeden z následujících efektů: 
- Grayscale
- Random dither
- Ordered dither
- C-dot dither
- D-dot dither
- Error diffusion
- Original

Každý efekt (kromě Original) dovoluje možnost 'enhance edges' (zvýraznění okrajů). Pro všechny efekty mimo Grayscale a Original se nejdřív aplikuje Grayscale. Uživatel může změnit nastavení efektu 'Error diffusion'. Uživatel také může upravený obrázek uložit/zahodit bez uložení a načíst nový obrázek.

## [2] UŽIVATELSKÉ ROZHRANÍ

UI aplikace má 3 hlavní části:

- **Prvotní 'Load Image' obrazovka**: První věc, kterou uživatel uvidí při zapnutí - výzva k načtení obrázku.

- **Ovládací panel**: Panel s tlačítky pro ovládání programu. Uživatel může z dropdown menu vybrat efekt, zaškrtnout, zda-li chce zvýraznit okraje, a kliknout na 'Apply effect' pro aplikaci efektu. Uživatel také může kliknout na tlačítko edit, kdy se otevře pop-up okno s dropdown menu. Zde může uživatel nastavit, jakým způsobem bude rozdělena chyba v efektu error diffusion.

- **Zobrazovací panel**: Panel zobrazující obrázek s aplikovaným efektem.

## [3] ZÁVISLOSTI (DEPENDENCIES)

Aplikace je závislá na verzi pythonu 3.7+
Závislosti vygenerované příkazem 'pip freeze':

customtkinter verze 5.2.1
darkdetect verze 0.8.0
networkx verze 3.2.1 
numpy verze 1.26.1
packaging verze 23.2
Pillow verze 10.1.0
scipy verze 1.11.3

Hlavní využití modulů:
- **CustomTkinter ([odkaz](https://customtkinter.tomschimansky.com/))**: Nadstavba standardního GUI modulu tkinter. Slouží pro hezčí GUI a Dark theme. Instalace: `pip install customtkinter`.

- **PIL ([odkaz](https://pillow.readthedocs.io/en/stable/))**: Python Image Library. Slouží k manipulaci s obrazem ve formátu ImageTk (pro Tkinter prvky). *Nebyl použit k aplikaci efektů. Instalace: `pip install Pillow`.

- **SciPy ([odkaz](https://scipy.org/))**: Open-source software for mathematics, science, and engineering. Společně s NumPy slouží ke zrychlení a optimalizaci algoritmů. Instalace: `pip install scipy`.

- **NumPy ([odkaz](https://numpy.org/))**: Viz SciPy. Instalace: `pip install numpy`.

Aplikace dále využívá tyto (snad) standardní moduly:

- random
- time - pro time wrapper měřící runtime funkcí
- packaging
- tkinter
