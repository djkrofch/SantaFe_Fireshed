// Composite an image collection and clip it to a boundary.
// See also: FilteredComposite, which filters the collection by bounds instead.
// Modified Dan Krofcheck - 12 - 13 - 17
// ------------------------------------


var calculateTasseledCap = function (image){
  var b = image.select("B2", "B3", "B4", "B5", "B6", "B7");
  //Coefficients are only for Landsat 8 TOA
	var brightness_coefficents= ee.Image([0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872]);
  var greenness_coefficents= ee.Image([-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608]);
  var wetness_coefficents= ee.Image([0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]);
  var fourth_coefficents= ee.Image([-0.8239, 0.0849, 0.4396, -0.058, 0.2013, -0.2773]);
  var fifth_coefficents= ee.Image([-0.3294, 0.0557, 0.1056, 0.1855, -0.4349, 0.8085]);
  var sixth_coefficents= ee.Image([0.1079, -0.9023, 0.4119, 0.0575, -0.0259, 0.0252]);

	var brightness = image.expression(
			'(B * BRIGHTNESS)',
			{
				'B':b,
				'BRIGHTNESS': brightness_coefficents
				}
			);
  var greenness = image.expression(
    '(B * GREENNESS)',
			{
				'B':b,
				'GREENNESS': greenness_coefficents
				}
			);
  var wetness = image.expression(
    '(B * WETNESS)',
			{
				'B':b,
				'WETNESS': wetness_coefficents
				}
			);
  var fourth = image.expression(
      '(B * FOURTH)',
        {
          'B':b,
          'FOURTH': fourth_coefficents
          }
        );
  var fifth = image.expression(
      '(B * FIFTH)',
        {
          'B':b,
          'FIFTH': fifth_coefficents
          }
        );
  var sixth = image.expression(
    '(B * SIXTH)',
    {
      'B':b,
      'SIXTH': sixth_coefficents
      }
    );
    
  brightness = brightness.reduce(ee.call("Reducer.sum"));
	greenness = greenness.reduce(ee.call("Reducer.sum"));
	wetness = wetness.reduce(ee.call("Reducer.sum"));
	fourth = fourth.reduce(ee.call("Reducer.sum"));
	fifth = fifth.reduce(ee.call("Reducer.sum"));
  sixth = sixth.reduce(ee.call("Reducer.sum"));
  var tasseled_cap = ee.Image(brightness)
  .addBands(greenness)
  .addBands(wetness)
  .addBands(fourth)
  .addBands(fifth)
  .addBands(sixth).rename(['brightness','greenness','wetness','fourth','fifth','sixth']);
  return tasseled_cap;
};

// This function masks clouds in Landsat 8 imagery.
var maskClouds = function(image) {
  var scored = ee.Algorithms.Landsat.simpleCloudScore(image);
  return image.updateMask(scored.select(['cloud']).lt(20));
};

// This function masks clouds and adds quality bands to Landsat 8 images.
var addQualityBands = function(image) {
  return maskClouds(image)
    // NDVI
    .addBands(image.normalizedDifference(['B5', 'B4']))
    // time in days
    .addBands(image.metadata('system:time_start'));
};

// Create a Landsat 8, median-pixel composite for Spring of 2000.
var collection = ee.ImageCollection('LANDSAT/LC8_L1T_TOA')
  .filterDate('2016-01-01', '2017-09-13')
  .map(addQualityBands)

Map.addLayer(SFFireshed);

var vizParams = {bands: ['B5', 'B4', 'B3'], min: 0, max: 0.4};

var qualityMosaic = collection.qualityMosaic('nd')
//var qualityLS = collection.qualityMosaic('nd');
var rockiesMosaic = qualityMosaic.clip(SFFireshed);
var rockiesTC = calculateTasseledCap(rockiesMosaic)
//console.log(rockies_LS8_TC);

Map.addLayer(SFFireshed);
Map.setCenter(-106, 34, 6);

// Export the image, specifying scale and region.
Export.image.toDrive({
  image: rockiesTC,
  description: 'SFFireshedTC',
  scale: 30,
  region: SFFireshed
});
