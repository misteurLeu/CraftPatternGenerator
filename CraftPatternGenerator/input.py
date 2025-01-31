# charts data are from https://github.com/maxcleme/beadcolors
COLORS_CHARTS = dict(
    Hama=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/hama.csv'
    ),
    HamaMini=dict(
        parent='Hama',
        avaible_keys=[
            'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10',
            'H11', 'H12', 'H17', 'H18', 'H20', 'H21', 'H22', 'H26', 'H27', 'H28',
            'H29', 'H30', 'H31', 'H43', 'H44', 'H45', 'H46', 'H47', 'H48', 'H49',
            'H60', 'H70', 'H71', 'H75', 'H76', 'H77', 'H78', 'H79', 'H82', 'H83',
            'H84', 'H95', 'H96', 'H97', 'H98'
        ],
        path=None
    ),
    Nabbi=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/nabbi.csv'
    ),
    Perler=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/perler.csv'
    ),
    PerlerMini=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/perler_mini.csv'
    ),
    PerlerCaps=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/perler_caps.csv'
    ),
    ArtkalA_2_6MM=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/artkal_a.csv'
    ),
    ArtkalC_2_6MM=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/artkal_c.csv'
    ),
    ArtkalR_5MM=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/artkal_r.csv'
    ),
    ArtkalS_5MM=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/artkal_s.csv'
    ),
    DiamondDotz=dict(
        parent=None,
        avaible_keys=None,
        path='https://beadcolors.eremes.xyz/raw/diamondDotz.csv'
    ),
)
COLORS_CHART_HEADER = ['ref', 'name', 'r', 'g', 'b', 'contributor']
COLORS_CODES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                'y', 'z']

preset_colors = {
    'transparent': (0, 0, 0, 0),
    'true_white': (255, 255, 255),
    'true_dark': (0, 0, 0),
    'white': (240, 240, 240),
    'black': (15, 15, 15),
    'mole': (91, 91, 91),
}
