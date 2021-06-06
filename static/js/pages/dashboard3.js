$(function () {

  'use strict';
    $.post('/map', {'country': "US"}, function(data) {

        document.getElementsByTagName('flights-map')[0].flights = JSON.parse(data.flightsArray)


    })


    $('#modalCountry').modal('hide');
    $('#modalCountry a').on('click', function () {

     $.post('/map', {'country': $(this).text()}, function(data) {

            document.getElementsByTagName('flights-map')[0].flights = JSON.parse(data.flightsArray)


        })
        $('#modalCountry').modal('hide');
    })

});
