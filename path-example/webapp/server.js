var express = require('express');
var app = express();

app.use(express.static('app'));
app.use('/tiles', express.static('../output'));
app.use('/meta', express.static('../output'));

var server = app.listen(8080, function () {
  var port = server.address().port;
  console.log('Salt Path example listening at http://localhost:%s', port);
});
