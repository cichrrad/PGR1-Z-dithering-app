
OBSAH============================================================================================================================================================================
=================================================================================================================================================================================

[1] Obecné informace
[2] Uživatelské rozhraní
[3] Závislosti (dependencies)

=================================================================================================================================================================================
=================================================================================================================================================================================



[1]OBECNÉ INFORMACE==============================================================================================================================================================
=================================================================================================================================================================================

Program DitherApp se skládá ze 4 .py souborů: 

	[DitherApp.py] 		= Hlavní soubor obsahující třídu root - aplikace je instance této třídy.
			 	  Také obsahuje funkce vykonávající základní operace (načtení/uložení).
			          Obsahuje algoritmy pro úpravu obrazu.
				  PRO ZAPNUTÍ APLIKACE JE TŘEBA RUNNOUT TENTO SOUBOR
	
	[ImageWidgets.py]	= Soubor obsahující prvky GUI vztahující se k prvotnímu načtení obrázku a
				  GUI prvek na kterém se vykresluje obrázek (Canvas).
				  
	[ControlWidgets.py]	= Soubor obsahující prvky GUI vztahující se k ovládání programu (levý ovlá-
				  dací sloupec s tlačítky). Logika / funkce přidávající prvkům funkcionalitu
				  jsou ale v DitherApp.py.

	[matrixDefinitions.py]	= Soubor obsahující různé matice pro Error diffusion.

Program Uživateli dovoluje načíst obrázek (ve standartních formátech) a aplikovat na něj jeden z následujících
efektů:  ["Grayscale",
	  "Random dither",
	  "Ordered dither",
	  "C-dot dither",
	  "D-dot dither",
	  "Error diffusion",
  	  "Original"],
kde každý efekt (kromě Original) dovoluje možnost 'enhance edges' (zvýraznění okrajů). Pro všechny efekty mimo
Grayscale a Original se nejdřív aplikuje Grayscale. Uživatel může změnit nastavení efektu 'Error diffusion'.
Uživatel také může upravený obrázek uložit / zahodit bez uložení a načíst nový obrázek.

=================================================================================================================================================================================
=================================================================================================================================================================================


[2]UŽIVATELKSÉ ROZHRANÍ==========================================================================================================================================================
=================================================================================================================================================================================

UI aplikace má 3 hlavní části: 

	[Prvotní 'Load Image' obrazovka] = První věc, kterou uživatel uvidí při zapnutí - výzva k načtení obrázku.

	[Ovládací panel] 		 = Panel s tlačítky pro ovládání programu. Uživatel může z dropdown menu vybrat efekt
			   		   ->zaškrtnout zda-li chce zvýraznit okraje-> kliknout na 'Apply effect' apikaci efektu.
					   Uživatel také může kliknout na tlačítko edit, kdy se otevře pop-up okno s dropdown menem.
					   Zde může uživatel nastavit jakým způsobem bude rozdělena chyba v efektu error diffusion.

	[Zobrazovací panel] 		 = Panel zobrazující obrázek s aplikovaným efektem
=================================================================================================================================================================================
=================================================================================================================================================================================



[3]DEPENDENCIES==================================================================================================================================================================
=================================================================================================================================================================================

Aplikace je závislá na verzi pythonu 3.7+
Závislosti vygenerované příkazem 'pip freeze':

customtkinter==5.2.1
darkdetect==0.8.0
networkx==3.2.1
numpy==1.26.1
packaging==23.2
Pillow==10.1.0
scipy==1.11.3

Hlavní využití modulů:

	[CustomTkinter ; https://customtkinter.tomschimansky.com/ ; pip install customtkinter] 	= nadstavba standardního GUI modulu tkinter. Slouží pro hezčí GUI a Dark theme.

	[PIL ; https://pillow.readthedocs.io/en/stable/ ; pip install Pillow ] 			= Python Image Library. Slouží k manipulaci s obrazem ve formátku ImageTk (pro Tkinter prvky).
										 		*Nebyl použit k aplikaci efektů
	
	[SciPy ; https://scipy.org/ ; pip install scipy] 					= Velmi populární modul pro výpočty -"open-source software for mathematics, science, and engineering".
												Společně s NumPy slouží ke zrychlení a optimalizaci algorimů - naivní přístup je často zakomentovaný,
												protože by trval moc dlouho (Python je pomalý :^/ )
	
	[NumPy ; https://numpy.org/ ; pip install numpy]					= viz SciPy
								
Aplikace dále využívá tyto (snad) standadtní moduly:

	[random] 
	[time] - pro time wrapper měřící runtime funkcí
	[packaging]
	[tkinter]

=================================================================================================================================================================================
=================================================================================================================================================================================

