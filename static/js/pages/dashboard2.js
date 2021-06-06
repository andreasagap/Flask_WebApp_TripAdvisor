$(function () {

  'use strict';

  var margin = {top: 10, right: 30, bottom: 30, left: 50},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;
        var svg = d3.select("#top10histogram")
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");
var x = d3.scaleLinear()
      .domain([-1, 1])     // can use this instead of 1000 to have the max of data: d3.max(data, function(d) { return +d.price })
      .range([0, width]);
  svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x));

  // Y axis: initialization
  var y = d3.scaleLinear()
      .range([height, 0]);
  var yAxis = svg.append("g")
$('#top10cityList a').on('click', function () {

  console.log($(this).text())
  d3.csv("/top10city/Sentiment Scores/"+$(this).text()+".csv")
  .row(function(d) {
        return {
            Value: d.Value,
        };
   })
  .get(function(data) {
   var headerNames = d3.keys(data[0]);

  // A function that builds the graph for a specific value of bin
  function update(nBin) {
      // width = $('#barchart').width - margin.left - margin.right

    // set the parameters for the histogram
    var histogram = d3.histogram()
        .value(function(d) { return d.Value; })   // I need to give the vector of value
        .domain(x.domain())  // then the domain of the graphic
        .thresholds(x.ticks(nBin)); // then the numbers of bins

    // And apply this function to data to get the bins
    var bins = histogram(data);

    // Y axis: update now that we know the domain
    y.domain([0, d3.max(bins, function(d) { return d.length; })]);   // d3.hist has to be called before the Y axis obviously
    yAxis
        .transition()
        .duration(1000)
        .call(d3.axisLeft(y));

    // Join the rect with the bins data
    var u = svg.selectAll("rect")
        .data(bins)

    // Manage the existing bars and eventually the new ones:
    u
        .enter()
        .append("rect") // Add a new rect for each new elements
        .merge(u) // get the already existing elements as well
        .transition() // and apply changes to all of them
        .duration(1000)
          .attr("x", 1)
          .attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; })
          .attr("width", function(d) { return x(d.x1) - x(d.x0) -1 ; })
          .attr("height", function(d) { return height - y(d.length); })
          .style("fill", "#ffffff")


    // If less bar in the new histogram, I delete the ones not in use anymore
    u
        .exit()
        .remove()

    }


  // Initialize with 5 bins
  update(4)
  });
  // X axis: scale and draw:

    $('#titleHistogram').html($(this).text())
    $('#modalCity').modal('hide');
})



    // get the data
    d3.csv("/top10city/Sentiment Scores/Athens, Greece.csv")
    .row(function(d) {
        return {
            Value: d.Value,
        };
   })
  .get(function(data) {


  // A function that builds the graph for a specific value of bin
  function update(nBin) {

      width = $('#barchart').offsetWidth - margin.left - margin.right

    // set the parameters for the histogram
    var histogram = d3.histogram()
        .value(function(d) { return d.Value; })   // I need to give the vector of value
        .domain(x.domain())  // then the domain of the graphic
        .thresholds(x.ticks(nBin)); // then the numbers of bins

    // And apply this function to data to get the bins
    var bins = histogram(data);

    // Y axis: update now that we know the domain
    y.domain([0, d3.max(bins, function(d) { return d.length; })]);   // d3.hist has to be called before the Y axis obviously
    yAxis
        .transition()
        .duration(1000)
        .call(d3.axisLeft(y));

    // Join the rect with the bins data
    var u = svg.selectAll("rect")
        .data(bins)

    // Manage the existing bars and eventually the new ones:
    u
        .enter()
        .append("rect") // Add a new rect for each new elements
        .merge(u) // get the already existing elements as well
        .transition() // and apply changes to all of them
        .duration(1000)
          .attr("x", 1)
          .attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; })
          .attr("width", function(d) { return x(d.x1) - x(d.x0) -1 ; })
          .attr("height", function(d) { return height - y(d.length); })
          .style("fill", "#ffffff")


    // If less bar in the new histogram, I delete the ones not in use anymore
    u
        .exit()
        .remove()

    }


  // Initialize with 5 bins
  update(4)
  });
});
