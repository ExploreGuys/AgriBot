# core/fertilizer_info.py

FERTILIZER_DATA = {
    'Urea': {
        'message': "The Nitrogen (N) value of your soil is low.",
        'suggestions': [
            "Add sawdust or fine woodchips to your soil – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up excess nitrogen.",
            "Plant heavy nitrogen feeding plants – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.",
            "Water – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.",
            "Sugar – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen in your soil.",
            "Add composted manure to the soil.",
            "Plant Nitrogen fixing plants like peas or beans.",
            "Use NPK fertilizers with high N value.",
            "Do nothing – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops."
        ]
    },
    'DAP': {
        'message': "The Phosphorous (P) value of your soil is low.",
        'suggestions': [
            "Add bone meal to the soil to increase phosphorous levels naturally.",
            "Apply rock phosphate, which is a slow-release source of phosphorous.",
            "Incorporate manure or compost, as they contain organic phosphorous.",
            "Check soil pH; phosphorous is most available to plants in slightly acidic to neutral soil (pH 6.0 to 7.0).",
            "Use NPK fertilizers with high P value like DAP (Diammonium Phosphate)."
        ]
    },
    '14-35-14': {
        'message': "Your soil requires a balanced boost with high Phosphorous.",
        'suggestions': [
            "Use 14-35-14 fertilizer to support strong root development and flowering.",
            "Ensure the soil has adequate moisture before applying granular fertilizer.",
            "Avoid over-application to prevent nutrient runoff into water sources."
        ]
    },
    '28-28': {
        'message': "Your soil needs an equal boost of Nitrogen and Phosphorous.",
        'suggestions': [
            "Apply 28-28-0 fertilizer during the early vegetative growth stage.",
            "This fertilizer is excellent for crops requiring rapid early growth.",
            "Monitor soil moisture as high-nitrogen fertilizers can cause 'burn' if the soil is too dry."
        ]
    },
    '10-26-26': {
        'message': "Your soil is low in Potassium (K) and Phosphorous (P).",
        'suggestions': [
            "Use 10-26-26 to improve crop quality and resistance to diseases.",
            "Potassium is vital for water regulation and enzyme activation in plants.",
            "Apply this fertilizer during the flowering or fruit-setting stage for best results."
        ]
    }
}