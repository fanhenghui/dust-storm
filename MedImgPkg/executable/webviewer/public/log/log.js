var log4js = require('log4js');

module.exports = {
    logger: function(name) {
      var logger = log4js.getLogger(name);
      logger.setLevel('INFO');
      return logger;
    },
    log4js: function() {
     return log4js;   
    }
}