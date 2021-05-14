/*
 * Author: Abdullah A Almsaeed
 * Date: 4 Jan 2014
 * Description:
 *      This is a demo file used only for the main dashboard (index.html)
 **/
function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}
function wordCloud(selector,words) {

    var fill = d3.scale.category20();

    //Construct the word cloud's SVG element
    var svg = d3.select(selector).append("svg")
        .attr("width", 600)
        .attr("height", 200)
        .append("g")
        var layout = d3.layout.cloud()
          .size([600, 200])
          .words(words)
          .padding(2)
      .rotate(0)
      .font("Impact")
      .fontSize(function(d) {
         return d.size;
      })
          .on("end", draw);
        layout.start();
    //Draw the word cloud
    function draw(words) {
       svg
      .attr("style", "display: block; margin-left: auto; margin-right: auto; background: white; text-align: center;")
      .append("g")
      .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
      .selectAll("text")
      .data(words)
      .enter().append("text")
      .style("font-size", function(d) { return d.size + "px"; })
      .style("font-family", "Impact")
      .style("fill", function(d, i) { return fill(i); })
      .attr("text-anchor", "middle")
      .attr("transform", function(d) {
        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
      })
      .text(function(d) { return d.text; });
    }
    function update() {
        svg.remove();

    }
}



$(function () {

  "use strict";

  //Make the dashboard widgets sortable Using jquery UI
  $(".connectedSortable").sortable({
    placeholder: "sort-highlight",
    connectWith: ".connectedSortable",
    handle: ".box-header, .nav-tabs",
    forcePlaceholderSize: true,
    zIndex: 999999
  });
  $(".connectedSortable .box-header, .connectedSortable .nav-tabs-custom").css("cursor", "move");

  //jQuery UI sortable for the todo list
  $(".todo-list").sortable({
    placeholder: "sort-highlight",
    handle: ".handle",
    forcePlaceholderSize: true,
    zIndex: 999999
  });
 $.post('/wordcloud', {'start': "2011-10-01"
          , 'end': '2021-03-01'}, function(data) {
            if(data.status == 200){
                // timestamp = new Date().getTime();
                var negative = JSON.parse(data.n)
                var positive = JSON.parse(data.p)
                var positive_text = []
                var negative_text = []
                   document.getElementById('positive_wordcloud').innerHTML = "";
                   document.getElementById('negative_wordcloud').innerHTML = "";
                Object.keys(positive).forEach(function(key) {
                  positive_text.push({text: key, size: 10 + positive[key] * 90})
                })
                Object.keys(negative).forEach(function(key) {
                  negative_text.push({text: key, size: 10 + negative[key] * 90})
                })

                wordCloud("#positive_wordcloud",positive_text)
                wordCloud("#negative_wordcloud",negative_text)
                document.getElementById('wordclouds').style.height = "550px";
                 document.getElementById("positive_wordcloud").style.margin = "12px";
                  document.getElementById("negative_wordcloud").style.margin = "12px";
               // var el = document.getElementById("wordcloud");
//                el.src = "/static/img/wordcloud.png?t=" + timestamp;


            }
            else{
                alert("Δεν βρέθηκαν reviews για την περίοδο " + start.format('YYYY-MM-DD') + " με " + end.format('YYYY-MM-DD'));
            }


          })
$(".bg-red").hover(function () {
    $(this).popover({
        title: "Groups of ages",
        content: "13-17: 2  <br />18-24: 60  <br />25-34: 306  <br />35-49: 550  <br />50-64: 550",
        html: true,
        placement: 'bottom'
    }).popover('show');
}, function () {
    $(this).popover('hide');
});
           $('[data-toggle="popover"]').popover()
  //bootstrap WYSIHTML5 - text editor
  $(".textarea").wysihtml5();
  $('.daterange').daterangepicker({
    ranges: {
    },
    startDate: moment('2011-10-01', 'YYYY-MM-DD'),
    endDate: moment('2021-03-01', 'YYYY-MM-DD')
  }, function (start, end) {
          $.post('/wordcloud', {'start': start.format('YYYY-MM-DD')
          , 'end': end.format('YYYY-MM-DD')}, function(data) {
            if(data.status == 200){
                // timestamp = new Date().getTime();
                var negative = JSON.parse(data.n)
                var positive = JSON.parse(data.p)
                var positive_text = []
                var negative_text = []
                   document.getElementById('positive_wordcloud').innerHTML = "";
                   document.getElementById('negative_wordcloud').innerHTML = "";
                Object.keys(positive).forEach(function(key) {
                  positive_text.push({text: key, size: 10 + positive[key] * 90})
                })
                Object.keys(negative).forEach(function(key) {
                  negative_text.push({text: key, size: 10 + negative[key] * 90})
                })

                wordCloud("#positive_wordcloud",positive_text)
                wordCloud("#negative_wordcloud",negative_text)
                document.getElementById('wordclouds').style.height = "550px";
                 document.getElementById("positive_wordcloud").style.margin = "12px";
                  document.getElementById("negative_wordcloud").style.margin = "12px";
               // var el = document.getElementById("wordcloud");
//                el.src = "/static/img/wordcloud.png?t=" + timestamp;


            }
            else{
                alert("Δεν βρέθηκαν reviews για την περίοδο " + start.format('YYYY-MM-DD') + " με " + end.format('YYYY-MM-DD'));
            }


          })
     })


  /* jQueryKnob */
  $(".knob").knob();

  //jvectormap data
  var visitorsData = {
    "US": 398, //USA
    "SA": 400, //Saudi Arabia
    "CA": 1000, //Canada
    "DE": 500, //Germany
    "FR": 760, //France
    "CN": 300, //China
    "AU": 700, //Australia
    "BR": 600, //Brazil
    "IN": 800, //India
    "GB": 320, //Great Britain
    "RU": 3000 //Russia
  };
  //World map by jvectormap
  $('#world-map').vectorMap({
    map: 'world_mill_en',
    backgroundColor: "transparent",
    regionStyle: {
      initial: {
        fill: '#e4e4e4',
        "fill-opacity": 1,
        stroke: 'none',
        "stroke-width": 0,
        "stroke-opacity": 1
      }
    },
    series: {
      regions: [{
        values: visitorsData,
        scale: ["#92c1dc", "#ebf4f9"],
        normalizeFunction: 'polynomial'
      }]
    },
    onRegionLabelShow: function (e, el, code) {
      if (typeof visitorsData[code] != "undefined")
        el.html(el.html() + ': ' + visitorsData[code] + ' new visitors');
    }
  });

  //Sparkline charts
  var myvalues = [1000, 1200, 920, 927, 931, 1027, 819, 930, 1021];
  $('#sparkline-1').sparkline(myvalues, {
    type: 'line',
    lineColor: '#92c1dc',
    fillColor: "#ebf4f9",
    height: '50',
    width: '80'
  });
  myvalues = [515, 519, 520, 522, 652, 810, 370, 627, 319, 630, 921];
  $('#sparkline-2').sparkline(myvalues, {
    type: 'line',
    lineColor: '#92c1dc',
    fillColor: "#ebf4f9",
    height: '50',
    width: '80'
  });
  myvalues = [15, 19, 20, 22, 33, 27, 31, 27, 19, 30, 21];
  $('#sparkline-3').sparkline(myvalues, {
    type: 'line',
    lineColor: '#92c1dc',
    fillColor: "#ebf4f9",
    height: '50',
    width: '80'
  });

  //The Calender
  $("#calendar").datepicker();

  //SLIMSCROLL FOR CHAT WIDGET
  $('#chat-box').slimScroll({
    height: '250px'
  });

  /* Morris.js Charts */
  // Sales chart
  var area = new Morris.Area({
    element: 'revenue-chart',
    resize: true,
    data: [
      {y: '2011 Q1', item1: 2666, item2: 2666},
      {y: '2011 Q2', item1: 2778, item2: 2294},
      {y: '2011 Q3', item1: 4912, item2: 1969},
      {y: '2011 Q4', item1: 3767, item2: 3597},
      {y: '2012 Q1', item1: 6810, item2: 1914},
      {y: '2012 Q2', item1: 5670, item2: 4293},
      {y: '2012 Q3', item1: 4820, item2: 3795},
      {y: '2012 Q4', item1: 15073, item2: 5967},
      {y: '2013 Q1', item1: 10687, item2: 4460},
      {y: '2013 Q2', item1: 8432, item2: 5713}
    ],
    xkey: 'y',
    ykeys: ['item1', 'item2'],
    labels: ['Item 1', 'Item 2'],
    lineColors: ['#a0d0e0', '#3c8dbc'],
    hideHover: 'auto'
  });
  var line = new Morris.Line({
    element: 'line-chart',
    resize: true,
    data: [
      {y: '2011 Q1', item1: 2666},
      {y: '2011 Q2', item1: 2778},
      {y: '2011 Q3', item1: 4912},
      {y: '2011 Q4', item1: 3767},
      {y: '2012 Q1', item1: 6810},
      {y: '2012 Q2', item1: 5670},
      {y: '2012 Q3', item1: 4820},
      {y: '2012 Q4', item1: 15073},
      {y: '2013 Q1', item1: 10687},
      {y: '2013 Q2', item1: 8432}
    ],
    xkey: 'y',
    ykeys: ['item1'],
    labels: ['Item 1'],
    lineColors: ['#efefef'],
    lineWidth: 2,
    hideHover: 'auto',
    gridTextColor: "#fff",
    gridStrokeWidth: 0.4,
    pointSize: 4,
    pointStrokeColors: ["#efefef"],
    gridLineColor: "#efefef",
    gridTextFamily: "Open Sans",
    gridTextSize: 10
  });

  //Donut Chart
  var donut = new Morris.Donut({
    element: 'sales-chart',
    resize: true,
    colors: ["#3c8dbc", "#f56954", "#00a65a"],
    data: [
      {label: "Download Sales", value: 12},
      {label: "In-Store Sales", value: 30},
      {label: "Mail-Order Sales", value: 20}
    ],
    hideHover: 'auto'
  });

  //Fix for charts under tabs
  $('.box ul.nav a').on('shown.bs.tab', function () {
    area.redraw();
    donut.redraw();
    line.redraw();
  });



});
