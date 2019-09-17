import React from "react"
import PropTypes from "prop-types"
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import { Link, Typography } from '@material-ui/core';
import Tooltip from '@material-ui/core/Tooltip';
import Zoom from '@material-ui/core/Zoom';
import ArrowForward from '@material-ui/icons/ArrowForward';
import IconButton from '@material-ui/core/IconButton';

import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles(theme => ({
  cardContent: {
    minHeight: 220,
    maxWidth: 360,
    maxHeight: 360,
  },
  cardTitle: {
    width: '80%',
  },
  descContent: {
    overflow: 'hidden',
    display: '-webkit-box',
    "-webkit-line-clamp": 4,
    "-webkit-box-orient": 'vertical',
  },
  card : {
    position: 'relative',
    boxShadow: '0 0 2px rgba(0, 0, 0, 0.2)',
    transition: 'all 0.25s',
    top: '0',
    height: '100%',
    '&:hover' : {
      boxShadow: '0 12px 16px rgba(55,55,55,.4)', 
      color: 'red',
      top: '-10px',
    },
  },
  moreLink: {
    marginLeft: 'auto',
    textDecoration: 'none',
    backgroundImage: 'none',
  },
  categoryBox: {
    position: 'absolute',
    display: 'inline',
    right: 0,
    top: 0,
    textTransform: 'uppercase',
    clipPath: 'polygon(0% 0%, 100% 0%, 100% 100%, 20% 100%)',
    backgroundColor: 'rgb(255, 229, 43)',
    padding: '5px 15px 5px 20px',
    margin: 0,
  }
}))

const Recipe = ({
  title,
  desc,
  url,
  category,
}) => {
  const classes = useStyles();

   //<Card className="shadow rounded">
  return (
    <Card className={classes.card}>
      <CardContent className={classes.cardContent}>
        <Typography className={classes.categoryBox} component="div" variant="h6" color="textSecondary">{category[0]}</Typography>
        <Typography gutterBottom className={classes.cardTitle} component="h1" variant="h6">
          {title}
        </Typography>
        <Tooltip enterDelay={1000} TransitionComponent={Zoom} TransitionProps={{ timeout: 300 }} title={desc}>
        <Typography variant="body2" color="textSecondary" component="p" className={classes.descContent} >
          {desc}
        </Typography>
        </Tooltip>
      </CardContent>
      <CardActions>
        <Link className={classes.moreLink} href={url}>
          <IconButton aria-label="show more"><ArrowForward/></IconButton>
        </Link>
      </CardActions>
    </Card>
  )
}

Recipe.propTypes = {
  title: PropTypes.string.isRequired,
  desc: PropTypes.string.isRequired,
  url: PropTypes.string.isRequired,
  category: PropTypes.string.isRequired,
}

export default Recipe
