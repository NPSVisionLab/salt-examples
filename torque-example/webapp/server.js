var express = require('express');
var app = express();

app.use(express.static('app'));
app.use('/tiles', express.static('../output'));

var server = app.listen(8081, function () {
  var port = server.address().port;
  console.log('Salt Torque example listening at http://localhost:%s', port);
});
